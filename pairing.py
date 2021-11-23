# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 14:35:57 2019

@author: prnvb
"""

import os
images = os.listdir('Dataset/Data/Images')

import pandas as pd
df = pd.DataFrame(columns = ['Name', 'Place', 'Year', 'Position', 'Ext'])

i = 0
for image in images:
    s = image.split('-')
    pos = s[-1].split('.')[0]
    ext = s[-1].split('.')[1]
    year = s[-2]
    name = ''
    for x in s[:-4]:
        name += x + '-'
    df.loc[i] = [image, name, year, pos, ext]
    i+=1
    
    
max_df = pd.DataFrame(columns=['Place', 'Max Year', 'Image'])
place_year = {}
place_image = {}
for i in range(0, 396):
    if (not df.iloc[i, 1] in place_year) or (df.iloc[i, 1] in place_year and place_year[df.iloc[i, 1]] < df.iloc[i, 2]):
        place_year[df.iloc[i, 1]] = df.iloc[i, 2]
        place_image[df.iloc[i, 1]] = df.iloc[i, 0]


        
ba = pd.DataFrame(columns = ['Before', 'After'])
for i in range(0, 396):
    if 'left' in df.iloc[i, 0]:
        ba.loc[i] = [df.iloc[i, 0], place_image[df.iloc[i, 1]]]
    else:
        ba.loc[i] = [df.iloc[i, 0], place_image[df.iloc[i, 1]].replace('left', 'right')]

beaf = {}
for i in range(0, 396):
    if(ba.iloc[i, 0] != ba.iloc[i, 1]):
        beaf[ba.iloc[i, 0]] = ba.iloc[i, 1]
        
import csv
with open('DataSet/Data/beforeAfter.csv', 'w') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['Before', 'After'])
    for key, value in beaf.items():
       writer.writerow([key, value])
       

check = pd.read_csv('DataSet/Data/beforeAfter.csv')
