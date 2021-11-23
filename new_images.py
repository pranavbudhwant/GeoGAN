# -*- coding: utf-8 -*-
"""
Created on Wed May  1 22:06:24 2019

@author: prnvb
"""

import pandas as pd
masks = pd.read_csv('DataSet/masks_199.csv')['External ID'].tolist()

import os
after_files = os.listdir('Clubbed/Pranav-SM/After/')

common = set(masks) & set(after_files)
uncommon = list( set(after_files).difference(common) )

from shutil import copyfile
src_dir = 'D:/dev/GeoGAN/Clubbed/Pranav-SM/After'
des_dir = 'D:/dev/GeoGAN/Clubbed/Pranav-SM/After/New'

for file in uncommon:
    copyfile( os.path.join(src_dir, file), os.path.join(des_dir, file) )