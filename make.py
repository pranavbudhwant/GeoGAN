# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 00:40:01 2019

@author: prnvb
"""

import cv2 as cv
import numpy as np
import pandas as pd
import os

def getXY():
    beaf = pd.read_csv('DataSet/Data/beforeAfter.csv')
    
    X = [] #List of numpy arrays -> 291x512x512x10
    Y = [] #List of numpy arrays -> 291x512x512x3
    
    for i in range(0, 291):
        before = cv.imread('DataSet/Data/Resized/Images/'+beaf.iloc[i, 0])
        after = cv.imread('DataSet/Data/Resized/Images/'+beaf.iloc[i, 1])
        before = (before - 127.5)/127.5
        after = (after - 127.5)/127.5
        #before = (2*(before - np.min(before))/(np.max(before) - np.min(before))) - 1
        #after = (2*(after - np.min(after))/(np.max(after) - np.min(after))) - 1
    
        after_masks = []
        for f in os.listdir('DataSet/Data/Resized/Masks'):
            mask = cv.imread(os.path.join('DataSet/Data/Resized/Masks/',f,beaf.iloc[i, 1]), cv.IMREAD_GRAYSCALE)
            #mask = (2*(mask - np.min(mask, axis=(0, 1)))/(np.max(mask, axis=(0, 1)) - np.min(mask, axis=(0, 1)))) - 1
            #mask = cv.normalize(mask, mask, 0, 1, cv.NORM_MINMAX)
            mask = (mask - 127.5)/127.5
            after_masks.append(mask)
            
        after_masks = np.asarray(after_masks)
        after_masks.resize((512, 512, 7))
        
        x = np.concatenate((before, after_masks), axis = -1)
        y = after
        
        X.append(x)
        Y.append(y)
    
    X = np.asarray(X)
    Y = np.asarray(Y)
    
    return X, Y


X, Y = getXY()

#Save Data
np.save('DataSet/Data/Resized/X_11Norm', X)
np.save('DataSet/Data/Resized/Y_11Norm', Y)

#Load Data:
X = np.load('DataSet/Data/Resized/X_11Norm.npy')
Y = np.load('DataSet/Data/Resized/Y_11Norm.npy')