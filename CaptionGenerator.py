#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 23:40:04 2020

@author: aayush
"""


from pickle import load
from numpy import argmax
from keras.preprocessing.sequence import pad_sequences
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.models import load_model
import numpy as np

def extract_features(filename):

    model = InceptionV3()
    model = Model(model.input, model.layers[-2].output)
    image = load_img(filename, target_size=(299, 299))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    feature = model.predict(image , verbose = 0)
    feature = feature.reshape(feature.shape[1] , 1)
    return feature


def word_to_index(i, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == i:
			return word
	return None


def generate_caption(model, tokenizer, photo, max_length):
	
    in_text = 'startseq'
    photo = np.reshape(photo , (1,2048))
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        ypred = model.predict([photo,sequence], verbose=0)
        ypred = argmax(ypred)
        word = word_to_index(ypred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text

s = '3651971126_309e6a5e22.jpg'
s1 = '3652584682_5b5c43e445.jpg'
tokenizer = load(open('tokenizer.pkl', 'rb'))
max_length = 34
model = load_model('weights/Model_deep_20.h5')
photo = extract_features("Flickr8k_Dataset/Flicker8k_Dataset/" + s)
#photo = extract_features("image.jpeg")
description = generate_caption(model, tokenizer, photo, max_length)
print(description)