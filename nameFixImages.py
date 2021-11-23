# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 22:57:49 2019

@author: prnvb
"""

import os
import re
images = os.listdir('Images/')

#Date Formats: YYYY-MM-DD or MM-DD-YYYY
image_names = {}
for image in images:
    name = image.split('_')[0].split('-')[0:-3]
    l = list(re.findall(r'[0-9]+-[0-9]+-[0-9]+-*[0-9]*', image)[0].split('-'))
    for i in range(0, len(l)):
        l[i] = format(l[i], '>02')
        if(len(l)==3):
            if(len(l[0])==4): #YYYY-MM-DD
                l[0], l[2] = l[2], l[0]
            elif(len(l[2])==4): #MM-DD-YYYY
                l[0], l[1], l[2] = l[1], l[0], l[2]
        else:
            if(len(l[1])==4):
                l[1], l[3] = l[3], l[1]
            elif(len(l[3])==4):
                l[1], l[2], l[3] = l[2], l[1], l[3]
    d = ''
    for k in l:
        d += k + '-'
    
    im = ''
    for x in name:
        im += x + '-'

    im += d + image.split('_')[1]
    image_names[image] = im

roots = []
for d in os.listdir('Masks'):
    roots.append('Masks/'+d)
roots.append('Images')

for root in roots:
    print(root)
    files = os.listdir(root)
    for f in files:
        os.rename(root+'/'+f, root+'/'+image_names[f])