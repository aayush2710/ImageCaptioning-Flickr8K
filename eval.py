#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 20:07:39 2020

@author: aayush
"""


from numpy import argmax
from pickle import load,dump
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu

def load_doc(filename):

	file = open(filename, 'r')
	text = file.read()
	file.close()
	return text


def load_set(filename):
	doc = load_doc(filename)
	dataset = list()
	for line in doc.split('\n'):
		if len(line) < 1:
			continue
		identifier = line.split('.')[0]
		dataset.append(identifier)
	return set(dataset)

def load_clean_captions(filename, dataset):
	doc = load_doc(filename)
	captions = dict()
	for line in doc.split('\n'):
		tokens = line.split()
		image_id, image_desc = tokens[0], tokens[1:]
		if image_id in dataset:
			# create list
			if image_id not in captions:
				captions[image_id] = list()
			desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
			captions[image_id].append(desc)
	return captions

def load_photo_features(filename, dataset):
	all_features = load(open(filename, 'rb'))
	features = {k: all_features[k] for k in dataset}
	return features

def to_lines(captions):
	all_desc = list()
	for key in captions.keys():
		[all_desc.append(d) for d in captions[key]]
	return all_desc

def create_tokenizer(captions):
	lines = to_lines(captions)
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer

def max_length(captions):
	lines = to_lines(captions)
	return max(len(d.split()) for d in lines)



def word_to_ix(i, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == i:
			return word
	return None


import numpy as np
def generate_caption(model, tokenizer, photo, max_length):

    in_text = 'startseq'

    for i in range(max_length):

        sequence = tokenizer.texts_to_sequences([in_text])[0]

        sequence = pad_sequences([sequence], maxlen=max_length)
        photo = np.reshape(photo , (1,2048))

        ypred = model.predict([photo,sequence], verbose=0)

        ypred = argmax(ypred)

        word = word_to_ix(ypred, tokenizer)

        if word is None:
            break

        in_text += ' ' + word

        if word == 'endseq':
            break
    return in_text


def evaluate_model(model, descriptions, photos, tokenizer, max_length):
	actual, predicted = list(), list()

	for key, caption_l in descriptions.items():

		ypred = generate_caption(model, tokenizer, photos[key], max_length)

		references = [d.split() for d in caption_l]
		actual.append(references)
		predicted.append(ypred.split())
	print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
	print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
	print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
	print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))


filename = 'Flickr8k_text/Flickr_8k.trainImages.txt'
train = load_set(filename)
train_captions = load_clean_captions('captions.txt', train)
tokenizer = create_tokenizer(train_captions)
vocab_size = len(tokenizer.word_index) + 1
max_length = max_length(train_captions)
filename = 'Flickr8k_text/Flickr_8k.testImages.txt'
test = load_set(filename)
test_descriptions = load_clean_captions('captions.txt', test)
test_features = load_photo_features('features.pkl', test)
filename = 'weights/Model_deep_20.h5'
model = load_model(filename)
evaluate_model(model, test_descriptions, test_features, tokenizer, max_length)
