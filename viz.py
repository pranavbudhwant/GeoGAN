# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 00:38:08 2019

@author: prnvb
"""

import os
import cv2 as cv
files = os.listdir('D:/dev/GeoGAN/DataSet/Data/Images/')

import random
img = random.randint(0, 396)
base = cv.imread('D:/dev/GeoGAN/DataSet/Data/Images/'+files[img])
comm = cv.imread('D:/dev/GeoGAN/DataSet/Data/Masks/Infra-Commercial/'+files[img], cv.IMREAD_GRAYSCALE)
fact = cv.imread('D:/dev/GeoGAN/DataSet/Data/Masks/Infra-Factory-Warehouse/'+files[img], cv.IMREAD_GRAYSCALE)
othe = cv.imread('D:/dev/GeoGAN/DataSet/Data/Masks/Infra-Other/'+files[img], cv.IMREAD_GRAYSCALE)
resi = cv.imread('D:/dev/GeoGAN/DataSet/Data/Masks/Infra-Residential/'+files[img], cv.IMREAD_GRAYSCALE)
road = cv.imread('D:/dev/GeoGAN/DataSet/Data/Masks/Road/'+files[img], cv.IMREAD_GRAYSCALE)
tree = cv.imread('D:/dev/GeoGAN/DataSet/Data/Masks/Trees/'+files[img], cv.IMREAD_GRAYSCALE)
watr = cv.imread('D:/dev/GeoGAN/DataSet/Data/Masks/Waterway/'+files[img], cv.IMREAD_GRAYSCALE)

import matplotlib.pyplot as plt
plt.figure()
plt.subplot(241).set_title('Base')
plt.subplot(241).axis('off')
plt.imshow(base)
plt.subplot(242).set_title('Commercial')
plt.subplot(242).axis('off')
plt.imshow(comm, cmap='gray')
plt.subplot(243).set_title('Factory')
plt.subplot(243).axis('off')
plt.imshow(fact, cmap='gray')
plt.subplot(244).set_title('Other')
plt.subplot(244).axis('off')
plt.imshow(othe, cmap='gray')
plt.subplot(245).set_title('Residential')
plt.subplot(245).axis('off')
plt.imshow(resi, cmap='gray')
plt.subplot(246).set_title('Roads')
plt.subplot(246).axis('off')
plt.imshow(road, cmap='gray')
plt.subplot(247).set_title('Trees')
plt.subplot(247).axis('off')
plt.imshow(tree, cmap='gray')
plt.subplot(248).set_title('Water')
plt.subplot(248).axis('off')
plt.imshow(watr, cmap='gray')