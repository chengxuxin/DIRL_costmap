import numpy as np
import tensorflow as tf
import os
import cv2 as cv


def get_queue(IMG_PATH):
    imgs = os.listdir(IMG_PATH)
    imgs.sort(key=lambda x: x[0:5])
    extracts = []
    for index in range(0, len(imgs), 1):  # set extraction interval
        extracts.append(imgs[index])
    return extracts


q = get_queue('/Users/David/Desktop/ao/img/')
print(q)
