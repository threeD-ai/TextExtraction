# -*- coding: utf-8 -*-
import sys
import os
import time
import concurrent.futures
import platform
import shutil

import pandas as pd
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

# self-identify which device(cpu/gpu) the code is currently running on.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# self-identify which operating system(Windows/Ubuntu) the code is currently running on.
if platform.system() == 'Windows':
    TESSERACT_PATH = "./packages/Tesseract-OCR/tesseract.exe"
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

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

"""
This is the main function to detect bounding boxes and subsequently run tesseract over it. 
It performs OCR and saves the result in the intermediary_folder
"""
def extraction(args):
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

    # run network in evaluation mode
    net.eval()

    with torch.no_grad():
        for image_tensors, image_path_list, aspect_ratios in demo_loader:
            batch_size = image_tensors.size(0)

            # send image tensors to device (cpu/gpu)
            if args.cpu:
                image_tensors = image_tensors.to('cpu')
            else:
                image_tensors = image_tensors.to(device)

            # forward pass
            y, _ = net(image_tensors)
            
            # run the loop over each item in the batch
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
                    df_file = result_folder + filename + '.csv'
                    image = process_utils.loadImage(image_path)
                    img = image[:,:,::-1]
                    with concurrent.futures.ThreadPoolExecutor(max_workers=7) as executor:
                        config = ("-l eng --oem 1 --psm 6")
                        texts = list(executor.map(lambda box: pytesseract.image_to_string(img[box[1]:box[3], box[0]:box[2]], config=config), boxes))

                        df_ocr = pd.DataFrame(boxes, columns = ['startX', 'startY', 'endX', 'endY'])
                        df_ocr['Text'] = texts
                        df_ocr.to_csv(df_file, index=False)

                # render results (optional)
                if args.render:

                    # Saving DETECTION results
                    detection_folder = args.detection_folder
    
                    # Save Character Mask and Link Mask in one file
                    render_img = np.hstack((score_text, score_link))
                    ret_score_text = process_utils.cvt2HeatmapImg(render_img)
                    image_path = image_path_list[i]
                    filename, _ = os.path.splitext(os.path.basename(image_path))
                    mask_file = detection_folder + filename + '_mask.jpg'
                    cv2.imwrite(mask_file, ret_score_text)

                    # Save image with all the bounding boxes planted on it
                    image = process_utils.loadImage(image_path)
                    img = image[:,:,::-1]
                    file_utils.saveResult(image_path, img, boxes, detection_folder, args.rotated_box)

                    # Saving RECOGNITION results

                    if args.recognize:
                        recognition_folder = args.recognition_folder

                        # make temp folder to store all the cropped snippets (of the image) as jpeg file
                        temp_folder = recognition_folder + '/temp/'
                        file_utils.make_clean_folder(temp_folder)

                        # excel file to store all the recognition results
                        excel_file = recognition_folder + filename + '_result.xlsx'

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

                        # loop over all the bounding boxes
                        i = 0
                        for box, text in zip(boxes,texts):
                            excelsheet.write(i+1,0,box[0])
                            excelsheet.write(i+1,1,box[1])
                            excelsheet.write(i+1,2,box[2])
                            excelsheet.write(i+1,3,box[3])
                            excelsheet.write(i+1,5,text)
                            excelsheet.write(i+1,6,'---')
                            excelsheet.write(i+1,7,'###')

                            #extract region of interest from the image
                            roi0 = img[box[1]:box[3], box[0]:box[2],:]

                            #resize the image to fit it in excel grid
                            try:
                                resized = process_utils.resize_height(roi0, 20)
                            except:
                                resized = roi0 

                            #saving image so that it can be inserted into excel sheet
                            cv2.imwrite(temp_folder +str(i+1) + '.jpg', resized)
                            excelsheet.insert_image(i+1, 4,(temp_folder +str(i+1) + '.jpg'), {'x_offset':3, 'y_offset':2, 'object_position':1})
                            i = i+1

                        excelbook.close()  
        
        # remove temp folder
        temp_folder = args.recognition_folder + '/temp/'
        if os.path.isdir(temp_folder):
            shutil.rmtree(temp_folder)

    print("\nTotal time for text extraction : {}s".format(round(time.time() - t, 1)))