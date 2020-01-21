#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 17:27:24 2020

@author: aayush
"""

import string

def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

def load_descriptions(filename):
    mapp = {}
    for i in doc.split('\n'):
        tokens = i.split()
        if len(i) < 2:
            continue
		
        image_id, image_desc = tokens[0], tokens[1:]

        image_id = image_id.split('.')[0]

        image_desc = ' '.join(image_desc)

        if image_id not in mapp:
            mapp[image_id] = list()

        mapp[image_id].append(image_desc)
    return mapp

def clean_descriptions(descriptions):

	table = str.maketrans('', '', string.punctuation)
	for key, desc_list in descriptions.items():
		for i in range(len(desc_list)):
			desc = desc_list[i]

			desc = desc.split()

			desc = [word.lower() for word in desc]

			desc = [w.translate(table) for w in desc]

			desc = [word for word in desc if len(word)>1]

			desc = [word for word in desc if word.isalpha()]

			desc_list[i] =  ' '.join(desc)

def to_vocabulary(descriptions):

	all_desc = set()
	for key in descriptions.keys():
		[all_desc.update(d.split()) for d in descriptions[key]]
	return all_desc

def save_descriptions(descriptions, filename):
	lines = list()
	for key, desc_list in descriptions.items():
		for desc in desc_list:
			lines.append(key + ' ' + desc)
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()
    
filename = 'Flickr8k_text/Flickr8k.token.txt'

doc = load_doc(filename)

captions = load_descriptions(doc)
print('Loaded: %d ' % len(captions))
clean_descriptions(captions)
vocabulary = to_vocabulary(captions)
print('Vocabulary Size: %d' % len(vocabulary))
save_descriptions(captions, 'captions.txt')
