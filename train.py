import numpy as np
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint

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







def data_generator(train_captions, photos, token, max_length, vocab_size , num_photos_per_batch):
    X1, X2, y = list(), list(), list()
    n=0

    while 1:
        for key, desc_list in train_captions.items():
            n+=1

            photo = photos[key][:,0]
            for desc in desc_list:
                seq = token.texts_to_sequences([desc])
                seq = seq[0]
                for i in range(1,len(seq)):
                    in_seq , op_seq = seq[:i],seq[i]
                    
                    in_seq = pad_sequences([in_seq],maxlen=max_length)[0]
                    
                    op_seq = to_categorical([op_seq],num_classes=vocab_size)[0]
                    X1.append(photo)
                    X2.append(in_seq)
                    y.append(op_seq)
            if n==num_photos_per_batch:
                yield [[np.array(X1), np.array(X2)], np.array(y)]
                X1, X2, y = list(), list(), list()
                n=0

def load_embeddings(path , tokens , vocab_size):
    
    embeddings_index = {} # empty dictionary
    f = open(path, encoding="utf-8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    dim = 300
    matrix = np.zeros((vocab_size ,dim) )
    for i in tokens.word_index:
        embedding_vector = embeddings_index.get(i)
        if embedding_vector is not None:
        
            matrix[tokens.word_index[i]-1] = embedding_vector
    
    return matrix
        
        
    
    


    
    

filename = 'Flickr8k_text/Flickr_8k.trainImages.txt'
train = load_set(filename)

train_captions = load_clean_captions('captions.txt', train)

train_features = load_photo_features('features.pkl', train)

tokenizer = create_tokenizer(train_captions)
vocab_size = len(tokenizer.word_index) + 1

max_length = max_length(train_captions)

embeddings = load_embeddings("glove6b/glove.6B.300d.txt" , tokenizer , vocab_size)


epochs = 20
steps = len(train_captions)
inputs1 = Input(shape=(2048,))
fe1 = Dropout(0.3)(inputs1)
fe3 = Dense(512 , activation = 'relu')(fe1)
fe4 = Dropout(0.3)(fe3)
fe2 = Dense(256, activation='relu')(fe4)
inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, 300, mask_zero=True , trainable = False)(inputs2)
se2 = Dropout(0.4)(se1)
se3 = LSTM(1024 , return_sequences=True)(se2)
se4 = Dropout(0.3)(se3)
se7 = LSTM(256)(se4)
decoder1 = add([fe2, se7])

decoder2 = Dense(512 ,activation='relu')(decoder1)
decoder3 = Dropout(0.5)(decoder2)
decoder4 = Dense(1024 ,activation='relu')(decoder3)
outputs = Dense(vocab_size, activation='softmax')(decoder4)
model = Model(inputs=[inputs1, inputs2], outputs=outputs)
model.layers[2].set_weights([embeddings])
model.layers[2].trainable = False

model.compile(loss='categorical_crossentropy', optimizer='adam' , metrics = ['accuracy'])


for i in range(epochs):

	generator = data_generator(train_captions, train_features, tokenizer, max_length, vocab_size , 1)
	model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
	model.save('model_' + str(i) + '.h5')
