# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
from pdf2image import convert_from_path
import platform
import shutil

# borrowed from https://github.com/lengstrom/fast-style-transfer/blob/master/src/utils.py
def get_files(img_dir):
    imgs, masks, xmls = list_files(img_dir)
    return imgs, masks, xmls

def make_clean_folder(path):
    if not os.path.isdir(path):
        os.mkdir(path)
    else:
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

def convert_pdf(input_folder, image_folder, name, pdfpath):
    counter = 0
    if platform.system() == 'Windows':
        pages = convert_from_path(input_folder + pdfpath, 300, poppler_path=r'./packages/poppler-0.68.0/bin')
    else:
        pages = convert_from_path(input_folder + pdfpath, 300)
    for page in pages:
        counter = counter + 1
        page.save(image_folder + name + "_%d.jpg" % (pages.index(page)), "JPEG")

def copy_img(input_folder, image_folder, path):
    _ = shutil.copyfile(input_folder + path, image_folder + path)

def copy_and_convert(input_folder, image_folder):
    for _, _, files in os.walk(input_folder):
        for filename in files:
            name, ext = os.path.splitext(filename)
            if (ext.lower() == '.pdf'):
                convert_pdf(input_folder, image_folder, name, filename)
            elif (ext.lower() == '.jpg' or ext.lower() == '.png' or ext.lower() == '.jpeg' or ext.lower() == '.tif'):
                copy_img(input_folder, image_folder, filename)

def list_files(in_path):
    img_files = []
    mask_files = []
    gt_files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == '.png' or ext == '.pgm':
                img_files.append(os.path.join(dirpath, file))
            elif ext == '.bmp':
                mask_files.append(os.path.join(dirpath, file))
            elif ext == '.xml' or ext == '.gt' or ext == '.txt':
                gt_files.append(os.path.join(dirpath, file))
            elif ext == '.zip':
                continue
    return img_files, mask_files, gt_files

def saveResult(img_file, img, boxes, dirname, rotated_box=False):
        """ save text detection result one by one
        Args:
            img_file (str): image file name
            img (array): raw image context
            boxes (array): array of result file
                Shape: [num_detections, 4] for BB output / [num_detections, 4] for QUAD output
        Return:
            None
        """
        img = np.array(img)

        # make result file list
        filename, file_ext = os.path.splitext(os.path.basename(img_file))

        # result directory
        res_file = dirname + "res_" + filename + '.txt'
        res_img_file = dirname + "res_" + filename + '.jpg'

        if not os.path.isdir(dirname):
            os.mkdir(dirname)

        if rotated_box:
            with open(res_file, 'w') as f:
                for i, box in enumerate(boxes):
                    poly = np.array(box).astype(np.int32).reshape((-1))
                    strResult = ','.join([str(p) for p in poly]) + '\r\n'
                    f.write(strResult)

                    poly = poly.reshape(-1, 2)
                    cv2.polylines(img, [poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)

        else:
            with open(res_file, 'w') as f:
                for box in boxes:
                    l,t,r,b = box
                    cv2.rectangle(img, (l,t), (r,b), (0, 0, 255), 2)
                    strResult = str(box)[1:-1]
                    f.write(strResult)


        # Save result image
        cv2.imwrite(res_img_file, img)

