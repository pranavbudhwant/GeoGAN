# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 10:58:49 2019

@author: prnvb
"""

root = 'D:\\dev\\GeoGAN\\DataSet\\Data\\'
import os

dirs = []
dirs.append(os.path.join(root,'Images'))
for d in os.listdir(os.path.join(root, 'Masks')):
    dirs.append(os.path.join(root,'Masks',d))

import cv2 as cv
for d in dirs:
    images = [x for x in os.listdir(d) if os.path.isfile(os.path.join(d,x))]
    for image in images:
        im = cv.imread(os.path.join(d, image))
        re = cv.resize(im, (512, 512))
        cv.imwrite(os.path.join(d,'Resized',image), re)