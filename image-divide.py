# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 21:49:42 2019

@author: prnvb
"""

import pandas as pd
import cv2 as cv
df = pd.read_csv('DataSet/masks_199.csv')

images = df.iloc[:,0].tolist()
root = 'Clubbed/AP/'
imroot = 'DataSet/Images/'

for image in images:
    img = cv.imread(root+image)
    print(img.shape)
    leftImage = img[:, 0:960, :]
    rightImage = img[:, 960:1920, :]
    name = image.split('.')[0]
    ext = image.split('.')[1]
    cv.imwrite(imroot+name+'_left.'+ext, leftImage)
    cv.imwrite(imroot+name+'_right.'+ext, rightImage)