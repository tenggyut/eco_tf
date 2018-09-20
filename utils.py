#encoding=utf8
import os
import random

import cv2
import sklearn.feature_extraction.image


def resize_keep_ratio(img, min_dim = 256):
    origin_h = img.shape[0]
    origin_w = img.shape[1]

    if origin_w < min_dim and origin_h >= min_dim:
        w = min_dim
        ratio = origin_h // origin_w
        h = w * ratio

    elif origin_h < min_dim and origin_w >= min_dim  :
        h = min_dim
        ratio = origin_w // origin_h
        w = h * ratio
    elif origin_h < min_dim and origin_w < min_dim:
        if origin_h < origin_w:
            h = min_dim
            ratio = origin_w // origin_h
            w = h * ratio
        else:
            w = min_dim
            ratio = origin_h // origin_w
            h = w * ratio
    else:
        h = origin_h
        w = origin_w

    res =  cv2.resize(img, (w, h))
    assert(min(res.shape[0], res.shape[1]) >= min_dim)
    return res

def crop_center(img,cropx,cropy):
    resized = resize_keep_ratio(img)
    y,x,c = resized.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy,startx:startx+cropx]

def random_crop(img, target_h, target_w):
    resized = resize_keep_ratio(img)
    patches = sklearn.feature_extraction.image.extract_patches_2d(resized, (target_h, target_w), 5)
    r = patches[random.randint(0, 4)]
    return r
