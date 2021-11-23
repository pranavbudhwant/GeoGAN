# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 15:43:41 2019

@author: prnvb
"""

import numpy as np
X = np.load('DataSet/Data/Resized/npy/OR/Shuffled/X_11Norm.npy')
Y = np.load('DataSet/Data/Resized/npy/OR/Shuffled/Y_11Norm.npy')

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)

np.save('DataSet/Data/Resized/npy/OR/Shuffled/X_train_11Norm.npy', X_train)
np.save('DataSet/Data/Resized/npy/OR/Shuffled/X_test_11Norm.npy', X_test)
np.save('DataSet/Data/Resized/npy/OR/Shuffled/Y_train_11Norm.npy', y_train)
np.save('DataSet/Data/Resized/npy/OR/Shuffled/Y_test_11Norm.npy', y_test)