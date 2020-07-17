# -*- coding: utf-8 -*-
import numpy as np
from skimage import io
import cv2
import math

def loadImage(img_file):
    img = io.imread(img_file)           # RGB order
    if img.shape[0] == 2: img = img[0]
    if len(img.shape) == 2 : img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:   img = img[:,:,:3]
    img = np.array(img)

    return img

def normalizeMeanVariance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    # should be RGB order
    img = in_img.copy().astype(np.float32)

    img -= np.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32)
    img /= np.array([variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0], dtype=np.float32)
    return img

def cvt2HeatmapImg(img):
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    return img

def getDetBoxes(textmap, linkmap, text_threshold, link_threshold, low_text, x_dilate=1, y_dilate=3, rotated_box=True):
    linkmap = linkmap.copy()
    textmap = textmap.copy()
    img_h, img_w = textmap.shape

    """ labeling method """
    ret, text_score = cv2.threshold(textmap, low_text, 1, 0)

    # vertical kernel
    center_x = int((x_dilate + 1) / 2)
    center_y = int((y_dilate + 1) / 2)

    inner = np.ones(center_x * center_y).reshape(center_y, center_x).astype(np.uint8)
    outer_r = np.zeros((x_dilate - center_x) * center_y).reshape(center_y, (x_dilate - center_x)).astype(np.uint8)
    outer_d = np.zeros((x_dilate) * -1 * (center_y - y_dilate)).reshape(y_dilate - center_y, x_dilate).astype(np.uint8)

    final = np.append(outer_r, inner, 1)
    Vkernel = np.append(outer_d, final, 0)

    text_score = cv2.dilate(text_score, Vkernel, 1)


    ret, link_score = cv2.threshold(linkmap, link_threshold, 1, 0)
    text_score_comb = np.clip(text_score + link_score, 0, 1)
    nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(text_score_comb.astype(np.uint8), connectivity=4)

    det = []

    for k in range(1,nLabels):
        # size filtering
        size = stats[k, cv2.CC_STAT_AREA]
        if size < 10: continue

        # thresholding (commented by me as thresholding didn't produce any significant difference in our case. Could be useful in very noisy images)
        if np.max(textmap[labels==k]) < text_threshold: continue

        # make segmentation map
        segmap = np.zeros(textmap.shape, dtype=np.uint8)
        segmap[labels==k] = 255
        segmap[np.logical_and(link_score==1, text_score==0)] = 0   # remove link area
        x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2.8)
        sx, ex, sy, ey = x - niter, x + w + niter + 1, y - niter, y + h + niter + 1
        # boundary check
        if sx < 0 : sx = 0
        if sy < 0 : sy = 0
        if ex >= img_w: ex = img_w
        if ey >= img_h: ey = img_h
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1 + niter, 1 + niter))
        segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)

        # make box
        np_contours = np.roll(np.array(np.where(segmap!=0)),1,axis=0).transpose().reshape(-1,2)

        if rotated_box:
            rectangle = cv2.minAreaRect(np_contours)
            box = cv2.boxPoints(rectangle)

            # align diamond-shape
            w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
            box_ratio = max(w, h) / (min(w, h) + 1e-5)
            if abs(1 - box_ratio) <= 0.1:
                l, r = min(np_contours[:,0]), max(np_contours[:,0])
                t, b = min(np_contours[:,1]), max(np_contours[:,1])
                box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)

        else:
            l, t, w, h = cv2.boundingRect(np_contours)
            box = np.array([l, t, l+w, t+h])

        # make clock-wise order
        # startidx = box.sum(axis=1).argmin()
        # box = np.roll(box, 4-startidx, 0)
        # box = np.array(box)
        det.append(box)

    # New Approach
    
    # box_list = []
    # Vcnts = cv2.findContours((text_score_comb*255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Vcnts = Vcnts[0] if len(Vcnts) == 2 else Vcnts[1]

    # for c in reversed(Vcnts):
    #     l,t,w,h = cv2.boundingRect(c)
    #     box_list.append(np.array([[l, t], [l+w, t], [l+w, t+h], [l, t+h]]))

    return det
    # return box_list

def adjustResultCoordinates(boxes, ratio_w, ratio_h, rotated_box, ratio_net = 2):
    if len(boxes) > 0:
        boxes = np.array(boxes)
        if rotated_box:
            for k in range(len(boxes)):
                if boxes[k] is not None:
                    boxes[k] *= (ratio_w * ratio_net, ratio_h * ratio_net)
        else:
            for k in range(len(boxes)):
                l,t,r,b = boxes[k]
                boxes[k] = [l*ratio_w*ratio_net, t*ratio_h * ratio_net, r*ratio_w * ratio_net,b*ratio_h * ratio_net]
                
    return boxes

def resize_height(img, height):
    ht, wd, c = img.shape
    ratio = height/ht
    width = int(ratio*wd)
    resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
    return resized
