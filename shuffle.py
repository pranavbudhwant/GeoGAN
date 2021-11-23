# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 08:27:15 2019

@author: prnvb
"""

import numpy as np
X = np.load('DataSet/Data/Resized/npy/OR/16/NoFlip/X16_11Norm_Train_Full.npy')
Y = np.load('DataSet/Data/Resized/npy/OR/16/NoFlip/Y16_11Norm_Train_Full.npy')

from sklearn.utils import shuffle
X_sh, Y_sh = shuffle(X, Y)

np.save('DataSet/Data/Resized/npy/OR/16/NoFlip/Shuffled/X16_11Norm_Train_Full.npy', X_sh)
np.save('DataSet/Data/Resized/npy/OR/16/NoFlip/Shuffled/Y16_11Norm_Train_Full.npy', Y_sh)