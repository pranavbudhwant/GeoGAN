# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 06:25:26 2019

@author: prnvb
"""

import cv2 as cv
import numpy as np
import pandas as pd
import os

beaf = pd.read_csv('DataSet/Data/beforeAfterTrain.csv')

images = []
for i in range(243):
    if beaf.iloc[i, 0] not in images:
        images.append(beaf.iloc[i, 0])
    if beaf.iloc[i, 1] not in images:
        images.append(beaf.iloc[i, 1])
    

X = []
Y = []
    
for i in range(len(images)):
    img = cv.imread('DataSet/Data/Resized/Images/'+images[i])
    img = (img - 127.5)/127.5
    
    zero_mask = np.zeros((512, 512, 4))
    x = np.concatenate((img, zero_mask), axis = -1)
    y = img    
    
    X.append(x)
    Y.append(y)

X = np.asarray(X, dtype=np.float16)
Y = np.asarray(Y, dtype=np.float16)

np.save('DataSet/Data/Resized/npy/OR/16/X16_NoMask_11Norm_Train', X)
np.save('DataSet/Data/Resized/npy/OR/16/Y16_NoMask_11Norm_Train', Y)