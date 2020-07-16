# -*- coding: utf-8 -*-
import sys
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from dataloader import RawDataset, AlignCollate
from PIL import Image

import cv2
from skimage import io
import numpy as np
import process_utils
import file_utils
import json
import zipfile

from detector import Detector

from collections import OrderedDict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    if not os.path.isdir(result_folder):
        os.mkdir(result_folder)
        
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
    demo_data = RawDataset(root=args.input_folder)
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
                boxes = process_utils.getDetBoxes(score_text, score_link, args.text_threshold, args.link_threshold, args.low_text)

                # coordinate adjustment
                ratio_h = ratio_w = 1 / aspect_ratios[i]
                boxes = process_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)

                # render results (optional)
                if args.render:
                    render_img = score_text.copy()
                    render_img = np.hstack((render_img, score_link))
                    ret_score_text = process_utils.cvt2HeatmapImg(render_img)

                    image_path = image_path_list[i]
                    filename, file_ext = os.path.splitext(os.path.basename(image_path))
                    mask_file = result_folder + "/res_" + filename + '_mask.jpg'
                    cv2.imwrite(mask_file, ret_score_text)
                    image = process_utils.loadImage(image_path)
                    file_utils.saveResult(image_path, image[:,:,::-1], boxes, dirname=result_folder)

    print("elapsed time : {}s".format(time.time() - t))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Text Extraction')

    """=================================== Part 1: DETECTION ==================================="""

    """ I/O Configuration """
    parser.add_argument('--input_folder', default='data/', type=str, help='folder path to input images')
    parser.add_argument('--ocr_result_folder', default='result/', type=str, help='folder path to output results')

    """ Data PreProcessing """
    parser.add_argument('--canvas_size', default=1920, type=int, help='image size for inference')
    parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')

    """ Data Loading Specifications """
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
    parser.add_argument('--batch_size', type=int, default=3, help='input batch size')

    """ Detection Model Specifications """
    parser.add_argument('--trained_model', default='weights/detector_mlt_25k.pth', type=str, help='pretrained model')
    parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
    parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
    parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')

    """ System Configuration """
    parser.add_argument('--cpu', default=False, action='store_true', help='Use cuda for inference')
    parser.add_argument('--verbose', default=False, action='store_true', help='show processing time and other details')
    parser.add_argument('--render', default=False, action='store_true', help='use to visualize intermediary results')

    args = parser.parse_args()

    test(args)

    
