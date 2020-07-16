# -*- coding: utf-8 -*-
import os
from natsort import natsorted
from skimage import io
from torch.utils.data import Dataset
import cv2
import numpy as np
import torch

class RawDataset(Dataset):

    def __init__(self, root):

        self.image_path_list = []
        for dirpath, dirnames, filenames in os.walk(root):
            for name in filenames:
                _, ext = os.path.splitext(name)
                ext = ext.lower()
                if ext == '.jpg' or ext == '.jpeg' or ext == '.png':
                    self.image_path_list.append(os.path.join(dirpath, name))

        self.image_path_list = natsorted(self.image_path_list)
        # print(self.image_path_list)
        self.nSamples = len(self.image_path_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):

        try:
            img = io.imread(self.image_path_list[index])           # RGB order
            if img.shape[0] == 2: img = img[0]
            if len(img.shape) == 2 : img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            if img.shape[2] == 4:   img = img[:,:,:3]
            img = np.array(img)
        except IOError:
            print(f'Corrupted image for {index}')

        return (img, self.image_path_list[index])

class AlignCollate(object):

    def __init__(self, square_size=1920, mag_ratio=1.5,  interpolation=cv2.INTER_LINEAR):
        self.square_size = square_size
        self.mag_ratio = mag_ratio
        self.interpolation = interpolation

    def resizeNormalize(self, in_img, max_h32, max_w32, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    # should be RGB order
        target_h, target_w, channel = in_img.shape
        
        resized = np.zeros((max_h32, max_w32, channel), dtype=np.float32)
        resized[0:target_h, 0:target_w, :] = in_img

        img = resized.copy()
        img -= np.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32)
        img /= np.array([variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0], dtype=np.float32)

        x = torch.from_numpy(img).permute(2, 0, 1)
        return x

    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)
        
        images, labels = zip(*batch)

        h_list = []
        w_list = [] 
        images_proc = []
        aspect_ratios = []
        for image in images:
            height, width, _ = image.shape
            target_size = self.mag_ratio * max(height, width)
            if target_size > self.square_size:
                target_size = self.square_size
            ratio = target_size / max(height, width)    
            aspect_ratios.append(ratio)
            target_h, target_w = int(height * ratio), int(width * ratio)
            proc = cv2.resize(image, (target_w, target_h), interpolation = self.interpolation)
            h_list.append(target_h)
            w_list.append(target_w)
        
            images_proc.append(proc)
        
        max_h = max(h_list)
        max_w = max(w_list)
        
        if max_h % 32 != 0:
            max_h = max_h + (32 - max_h % 32)
        if max_w % 32 != 0:
            max_w = max_w + (32 - max_w % 32)
        
        image_tensors = [self.resizeNormalize(image, max_h, max_w) for image in images_proc]
        image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)

        return image_tensors, labels, aspect_ratios