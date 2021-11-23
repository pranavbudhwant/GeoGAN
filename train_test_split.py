# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 11:27:13 2019

@author: prnvb
"""

import pandas as pd

beaf = pd.read_csv('DataSet/Data/beforeAfter.csv')
test = beaf.iloc[101:139, :]

test = test.append(beaf.iloc[281:,:])

train = beaf.iloc[:101, :]
train = train.append(beaf.iloc[139:281, :])

train.to_csv('DataSet/Data/beforeAfterTrain.csv', index=False)
test.to_csv('DataSet/Data/beforeAfterTest.csv', index=False)

x = pd.read_csv('DataSet/Data/beforeAfterTest.csv')