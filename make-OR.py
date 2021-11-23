# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 20:50:09 2019

@author: prnvb
"""
import cv2 as cv
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

def getX_train(dt=np.float16):
    beaf = pd.read_csv('DataSet/Data/beforeAfterTrain.csv')
    X = [] #List of numpy arrays -> 291x512x512x10
    for i in range(0, 243):
        before = cv.imread('DataSet/Data/Resized/Images/'+beaf.iloc[i, 0])
        before = (before - 127.5)/127.5
        
        mask_new = np.zeros((512, 512))
        mask = np.zeros((512, 512))

        for f in os.listdir('DataSet/Data/Resized/Masks')[:4]:
            mask = cv.imread(os.path.join('DataSet/Data/Resized/Masks/',f,beaf.iloc[i, 1]), cv.IMREAD_GRAYSCALE)
            mask = np.where(mask>0, 1, 0)
            mask_new = np.logical_or(mask, mask_new).astype(float)
       
        after_masks = (mask_new)
        after_masks = np.expand_dims(after_masks, axis = 2) 

        for f in os.listdir('DataSet/Data/Resized/Masks')[4:]:
            mask = cv.imread(os.path.join('DataSet/Data/Resized/Masks/',f,beaf.iloc[i, 1]), cv.IMREAD_GRAYSCALE)
            mask = np.where(mask>0, 1, 0)
            mask = np.expand_dims(mask, axis = 2) 
            after_masks = np.concatenate((after_masks, mask), axis = -1)

        x = np.concatenate((before, after_masks), axis = -1)

        X.append(x)
        X.append(np.fliplr(x))
        print(i)
        #X.append(np.flipud(x))
        #X.append(np.fliplr(np.flipud(x)))

    X = np.asarray(X, dtype=dt)
    return X

np.save('DataSet/Data/Resized/npy/OR/16/HFlip/X16_11Norm_Train', X)

def getX_test(dt=np.float16):
    beaf = pd.read_csv('DataSet/Data/beforeAfterTest.csv')
    X = [] #List of numpy arrays -> 291x512x512x10
    for i in range(0, 48):
        before = cv.imread('DataSet/Data/Resized/Images/'+beaf.iloc[i, 0])
        before = (before - 127.5)/127.5
        
        mask_new = np.zeros((512, 512))
        mask = np.zeros((512, 512))

        for f in os.listdir('DataSet/Data/Resized/Masks')[:4]:
            mask = cv.imread(os.path.join('DataSet/Data/Resized/Masks/',f,beaf.iloc[i, 1]), cv.IMREAD_GRAYSCALE)
            mask = np.where(mask>0, 1, 0)
            mask_new = np.logical_or(mask, mask_new).astype(float)
       
        after_masks = (mask_new)
        after_masks = np.expand_dims(after_masks, axis = 2) 

        for f in os.listdir('DataSet/Data/Resized/Masks')[4:]:
            mask = cv.imread(os.path.join('DataSet/Data/Resized/Masks/',f,beaf.iloc[i, 1]), cv.IMREAD_GRAYSCALE)
            mask = np.where(mask>0, 1, 0)
            mask = np.expand_dims(mask, axis = 2) 
            after_masks = np.concatenate((after_masks, mask), axis = -1)

        x = np.concatenate((before, after_masks), axis = -1)

        X.append(x)
        
    X = np.asarray(X, dtype=dt)
    return X

np.save('DataSet/Data/Resized/npy/OR/X16_11Norm_Test', X)

def getY_train(dt=np.float16):
    beaf = pd.read_csv('DataSet/Data/beforeAfterTrain.csv')
    Y = [] #List of numpy arrays -> 291x512x512x3
    for i in range(0, 243):
        after = cv.imread('DataSet/Data/Resized/Images/'+beaf.iloc[i, 1])
        
        after = (after- 127.5)/127.5

        y = after
        Y.append(y)
        Y.append(np.fliplr(y))
        #Y.append(np.flipud(y))
        #Y.append(np.fliplr(np.flipud(y)))
        
        print(i)
        
    Y = np.asarray(Y, dtype=dt)
    return Y

np.save('DataSet/Data/Resized/npy/OR/16/HFlip/Y16_11Norm_Train', Y)

def getY_test(dt=np.float16):
    beaf = pd.read_csv('DataSet/Data/beforeAfterTest.csv')
    Y = [] #List of numpy arrays -> 291x512x512x3
    for i in range(0, 48):
        after = cv.imread('DataSet/Data/Resized/Images/'+beaf.iloc[i, 1])
        
        after = (after- 127.5)/127.5

        y = after
        Y.append(y)
        
        print(i)
        
    Y = np.asarray(Y, dtype=dt)
    return Y

np.save('DataSet/Data/Resized/npy/OR/Y16_11Norm_Test', Y)


def getXY(dt=np.float64):
    beaf = pd.read_csv('DataSet/Data/beforeAfter.csv')
    X = [] #List of numpy arrays -> 291x512x512x10
    Y = [] #List of numpy arrays -> 291x512x512x3
    for i in range(0, 291):
        before = cv.imread('DataSet/Data/Resized/Images/'+beaf.iloc[i, 0])
        after = cv.imread('DataSet/Data/Resized/Images/'+beaf.iloc[i, 1])
        
        #before = cv.cvtColor(before, cv.COLOR_BGR2RGB)
        #after = cv.cvtColor(after, cv.COLOR_BGR2RGB)
        
        before = (before - 127.5)/127.5
        after = (after- 127.5)/127.5
        
        #before = cv.normalize(before, before, 0, 1, cv.NORM_MINMAX)
        #after = cv.normalize(after, after, 0, 1, cv.NORM_MINMAX)
    
        mask_new = np.zeros((512, 512))
        mask = np.zeros((512, 512))

        for f in os.listdir('DataSet/Data/Resized/Masks')[:4]:
            mask = cv.imread(os.path.join('DataSet/Data/Resized/Masks/',f,beaf.iloc[i, 1]), cv.IMREAD_GRAYSCALE)
            mask = np.where(mask>0, 1, 0)
            mask_new = np.logical_or(mask, mask_new).astype(float)
        
        #plt.figure()
        #plt.subplot(121)
        #plt.imshow(((after+1)/2)*255)
        #plt.subplot(122)        
        #plt.imshow(mask_new, cmap='gray')

        #mask_new = cv.normalize(mask_new, mask_new, 0, 1, cv.NORM_MINMAX)
        after_masks = (mask_new)
        after_masks = np.expand_dims(after_masks, axis = 2) 
        #plt.imshow(after_masks[:, :, 0], cmap='gray')

        for f in os.listdir('DataSet/Data/Resized/Masks')[4:]:
            mask = cv.imread(os.path.join('DataSet/Data/Resized/Masks/',f,beaf.iloc[i, 1]), cv.IMREAD_GRAYSCALE)
            mask = np.where(mask>0, 1, 0)
            mask = np.expand_dims(mask, axis = 2) 

            #mask = cv.normalize(mask, mask, 0, 1, cv.NORM_MINMAX)
            #after_masks.append(mask.astype(float))
            after_masks = np.concatenate((after_masks, mask), axis = -1)
            
        #after_masks = np.asarray(after_masks, dtype=float)
        #after_masks = after_masks.reshape((512, 512, 4))
        
        #plt.imshow(after_masks[:, :, 0], cmap='gray')        
        
        x = np.concatenate((before, after_masks), axis = -1)
        y = after
        
        X.append(x)
        X.append(np.fliplr(x))
        X.append(np.flipud(x))
        X.append(np.fliplr(np.flipud(x)))
        
        Y.append(y)
        Y.append(np.fliplr(y))
        Y.append(np.flipud(y))
        Y.append(np.fliplr(np.flipud(y)))
        
        print(i)
    
    X = np.asarray(X, dtype=dt)
    Y = np.asarray(Y, dtype=dt)
    
    return X, Y


X, Y = getXY()


k = 657

plt.figure()
plt.subplot(231)
plt.imshow(((X[k, :, :, 0:3]+1)/2)*255)

plt.subplot(232)
plt.imshow(((Y[k, :, :, :]+1)/2)*255)

plt.subplot(233)
plt.imshow(X[k, :, :, 3], cmap='gray')

plt.subplot(234)
plt.imshow(X[k, :, :, 4], cmap='gray')

plt.subplot(235)
plt.imshow(X[k, :, :, 5], cmap='gray')

plt.subplot(236)
plt.imshow(X[k, :, :, 6], cmap='gray')


#Save Data
np.save('DataSet/Data/Resized/npy/OR/X16_11Norm', X)
np.save('DataSet/Data/Resized/npy/OR/Y16_11Norm', Y)

#Load Data:
X = np.load('DataSet/Data/Resized/npy/OR/16/X16_11Norm.npy').astype(np.float32)
Y = np.load('DataSet/Data/Resized/npy/OR/16/Y16_11Norm.npy').astype(np.float32)