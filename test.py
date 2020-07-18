# -*- coding: utf-8 -*-
import sys
import os
import time
import argparse
import concurrent.futures
import platform

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import json
import zipfile
from PIL import Image
import pytesseract
import cv2
from skimage import io
import numpy as np
import xlsxwriter as xw

import process_utils
import file_utils
from dataloader import RawDataset, AlignCollate
from detector import Detector

from collections import OrderedDict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if platform.system() == 'Windows':
    TESSERACT_PATH = "./packages/Tesseract-OCR/tesseract.exe"

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

def test(args):
    t = time.time()
    result_folder = args.ocr_result_folder
    
    # load Detection network
    net = Detector()     # initialize

    # load Detection trained model
    print('Loading weights from checkpoint (' + args.trained_model + ')')
    if args.cpu:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))
    else:
        net = torch.nn.DataParallel(net).to(device)
        net.load_state_dict(torch.load(args.trained_model, map_location=device))
        cudnn.benchmark = False

    # define dataset and dataloader
    AlignCollate_demo = AlignCollate(square_size = args.canvas_size, mag_ratio=args.mag_ratio)
    demo_data = RawDataset(root=args.image_folder)
    demo_loader = torch.utils.data.DataLoader(
        demo_data, batch_size=args.batch_size,
        shuffle=False,
        num_workers=int(args.workers),
        collate_fn=AlignCollate_demo, pin_memory=True)

    net.eval()

    with torch.no_grad():
        for image_tensors, image_path_list, aspect_ratios in demo_loader:
            batch_size = image_tensors.size(0)
            if args.cpu:
                image_tensors = image_tensors.to('cpu')
            else:
                image_tensors = image_tensors.to(device)

            y, feature = net(image_tensors)

            for i in range(batch_size):

                # make score and link map               
                score_text = y[i,:,:,0].cpu().data.numpy()
                score_link = y[i,:,:,1].cpu().data.numpy()
            
                # Post-processing
                boxes = process_utils.getDetBoxes(score_text, score_link, args.text_threshold, args.link_threshold, args.low_text, args.x_dilate, args.y_dilate, args.rotated_box)

                # coordinate adjustment
                ratio_h = ratio_w = 1 / aspect_ratios[i]
                boxes = process_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h, args.rotated_box)

                # Recognition Part
                if args.recognize:
                    image_path = image_path_list[i]
                    filename, _ = os.path.splitext(os.path.basename(image_path))
                    numpy_file = result_folder + "/" + filename + '.npy'
                    image = process_utils.loadImage(image_path)
                    img = image[:,:,::-1]
                    with concurrent.futures.ThreadPoolExecutor(max_workers=7) as executor:
                        config = ("-l eng --oem 1 --psm 6")
                        texts = list(executor.map(lambda box: pytesseract.image_to_string(img[box[1]:box[3], box[0]:box[2]], config=config), boxes))
                        data_tuples = list(zip(boxes, texts))
                        np.save(numpy_file, np.array(data_tuples))

                # render results (optional)
                if args.render:
                    render_img = score_text.copy()
                    render_img = np.hstack((render_img, score_link))
                    ret_score_text = process_utils.cvt2HeatmapImg(render_img)

                    image_path = image_path_list[i]
                    filename, _ = os.path.splitext(os.path.basename(image_path))
                    mask_file = result_folder + "/res_" + filename + '_mask.jpg'
                    cv2.imwrite(mask_file, ret_score_text)

                    image = process_utils.loadImage(image_path)
                    img = image[:,:,::-1]
                    file_utils.saveResult(image_path, img, boxes, dirname=result_folder)

                    if args.recognize:
                        temp_folder = result_folder + '/temp/'
                        file_utils.make_clean_folder(temp_folder)

                        excel_file = result_folder + "/" + filename + '_result.xlsx'

                        excelbook = xw.Workbook(excel_file)
                        excelsheet = excelbook.add_worksheet('Sheet1')
                        excelsheet.set_column(4, 4, 31)
                        excelsheet.set_column(5, 5, 25)
                        excelsheet.set_column(6, 7, 20)
                        excelsheet.set_default_row(15)

                        bold = excelbook.add_format({'bold': True})
                        excelsheet.write(0,0, 'start_X', bold)
                        excelsheet.write(0,1, 'start_Y', bold)
                        excelsheet.write(0,2, 'end_X', bold)
                        excelsheet.write(0,3, 'end_Y', bold)
                        excelsheet.write(0,5, 'Text', bold)
                        excelsheet.write(0,4, 'Image', bold)
                        excelsheet.write(0,6, 'Ground_Truth', bold)
                        excelsheet.write(0,7, 'Label', bold)
                        i = 0
                        for box, text in data_tuples:
                            excelsheet.write(i+1,0,box[0])
                            excelsheet.write(i+1,1,box[1])
                            excelsheet.write(i+1,2,box[2])
                            excelsheet.write(i+1,3,box[3])
                            excelsheet.write(i+1,5,text)
                            excelsheet.write(i+1,6,'---')
                            excelsheet.write(i+1,7,'###')
                            roi0 = img[box[1]:box[3], box[0]:box[2],:]
                            try:
                                resized = process_utils.resize_height(roi0, 20)
                            except:
                                resized = roi0 
                            cv2.imwrite(temp_folder +str(i+1) + '.jpg', resized)
                            excelsheet.insert_image(i+1, 4,(temp_folder +str(i+1) + '.jpg'), {'x_offset':3, 'y_offset':2, 'object_position':1})
                            i = i+1
                        excelbook.close()  

    print("elapsed time : {}s".format(time.time() - t))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Text Extraction')

    """=================================== Part 1: DETECTION ==================================="""

    """ I/O Configuration """
    parser.add_argument('--input_folder', default='data/', type=str, help='folder path to input files')
    parser.add_argument('--image_folder', default='img/', type=str, help='folder path to input (converted) images')
    parser.add_argument('--ocr_result_folder', default='result/', type=str, help='folder path to output results')

    """ Data PreProcessing """
    parser.add_argument('--canvas_size', default=1920, type=int, help='image size for inference')
    parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')

    """ Data Loading Specifications """
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
    parser.add_argument('--batch_size', type=int, default=3, help='input batch size')

    """ Detection Model Specifications """
    parser.add_argument('--trained_model', default='weights/detector_mlt_25k.pth', type=str, help='pretrained model')
    parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold') #This threshold is not used in our case.
    parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score') #0.003 was used with 0.2-tt and 0.1-lt
    parser.add_argument('--link_threshold', default=0.1, type=float, help='link confidence threshold')
    parser.add_argument('--rotated_box', default=False, action='store_true', help='use this to get rotated rectangles (bounding box)') # Currently not handling for rotated boxes
    parser.add_argument('--x_dilate', default=1, type=int, help='left x-padding during post processing')
    parser.add_argument('--y_dilate', default=3, type=int, help='up y-padding during post processing')

    """ System Configuration """
    parser.add_argument('--cpu', default=False, action='store_true', help='Use cuda for inference')
    parser.add_argument('--verbose', default=False, action='store_true', help='show processing time and other details')
    parser.add_argument('--render', default=False, action='store_true', help='use to visualize intermediary results')

    """=================================== Part 2: RECOGNITION ==================================="""

    # parser.add_argument('--image_folder', default='data/', type=str, help='path to image_folder which contains text images')
    parser.add_argument('--recognize', default=False, action='store_true', help='flag to enable recognition')
    # parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    # parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    # parser.add_argument('--saved_model', required=True, help="path to saved_model to evaluation")
    # """ Data processing """
    # parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    # parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    # parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    # parser.add_argument('--rgb', action='store_true', help='use rgb input')
    # parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    # parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    # parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    # """ Model Architecture """
    # parser.add_argument('--Transformation', type=str, required=True, help='Transformation stage. None|TPS')
    # parser.add_argument('--FeatureExtraction', type=str, required=True, help='FeatureExtraction stage. VGG|RCNN|ResNet')
    # parser.add_argument('--SequenceModeling', type=str, required=True, help='SequenceModeling stage. None|BiLSTM')
    # parser.add_argument('--Prediction', type=str, required=True, help='Prediction stage. CTC|Attn')
    # parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    # parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    # parser.add_argument('--output_channel', type=int, default=512,
    #                     help='the number of output channel of Feature extractor')
    # parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

    args = parser.parse_args()

    file_utils.make_clean_folder(args.image_folder)
    file_utils.make_clean_folder(args.ocr_result_folder)
    
    file_utils.copy_and_convert(args.input_folder, args.image_folder)
    test(args)