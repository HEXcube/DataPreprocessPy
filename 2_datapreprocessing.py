#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 4 10:51:02 2018

@author: team_pyrastinators
"""

import random
import numpy as np
from sklearn import preprocessing
randoms = random.sample(range(-100, 100), 5)
wrapper_list = []
wrapper_list.append(randoms)
data = np.array(wrapper_list)
#data = np.array([[7.3,-9.9,-4.5]])
# data = np.array([[5.1,-2.9,3.3],
#                 [-1.2,7.8,-6.1],
#                 [3.9,0.4,2.1],
#                 [7.3,-9.9,-4.5]])
print(data)

#binarization - convert numerical values into boolean values

#binarized_data = preprocessing.Binarizer(theshold=2.1).transform(data)
print("***********************Binarization*************************")
binarized_data = preprocessing.binarize(data,threshold=2.1)
print(binarized_data)

print("***********************Scaling*******************************")
#Scaling - Scaled data has zero mean and unit variance
scaled_data = preprocessing.scale(data)
print(scaled_data)

print("***********************Minmax_Scaling*************************")
scaler0_1 = preprocessing.minmax_scale(data,feature_range=(0,1),axis=0)
print(scaler0_1)

print("***********************Normalization************************")
#normalization - process of scaling individual samples to have unit norm

normalized_data_1 = preprocessing.normalize(data, norm = 'l1')
normalized_data_2= preprocessing.normalize(data, norm ='l2')
print(normalized_data_1)
print(normalized_data_2)
print("***********************Mean_removal**********************")

#mean removal
print("mean before meanremoval ", data.mean(axis=0))
print("mean before meanremoval ", data.std(axis=0))

print("mean after meanremoval ", scaled_data.mean(axis=0))
print("mean after meanremoval ", scaled_data.std(axis=0))
