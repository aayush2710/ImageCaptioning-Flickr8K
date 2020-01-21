#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 16:36:31 2020

@author: aayush
"""

from os import listdir
from pickle import dump
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.inception_v3 import preprocess_input
import numpy as np
from keras.models import Model

model = InceptionV3()
model = Model(model.input, model.layers[-2].output)
i=1
features = {}
for k in listdir("Flickr8k_Dataset/Flicker8k_Dataset/"):
    fname = "Flickr8k_Dataset/Flicker8k_Dataset/" + k
    image = load_img(fname , target_size=(299, 299))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    feature = model.predict(image , verbose = 0)
    feature = feature.reshape(feature.shape[1] , 1)
    features[k.split(".")[0]] = feature
    
    if(i%100==0):
        print(i)
    i+=1
    
dump(features, open('features.pkl', 'wb'))
    
    
    
    
