# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 20:49:02 2019

@author: prnvb
"""

import os
import cv2 as cv

roots = []
for d in os.listdir('Dataset/Masks'):
    roots.append('Dataset/Masks/'+d)

for root in roots:
    print(root)
    files = [x for x in os.listdir(root) if os.path.isfile(root+'/'+x)]
    for f in files:
        path = os.path.join(root, f)
        newp = os.path.join(root, 'Split')
        image = cv.imread(path)
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        leftImage = gray[:,0:960]
        rightImage = gray[:,960:1920]
        name = f.split('.')[0]
        ext = f.split('.')[1]
        cv.imwrite(os.path.join(newp, name+'_left.'+ext), leftImage)
        cv.imwrite(os.path.join(newp, name+'_right.'+ext), rightImage)