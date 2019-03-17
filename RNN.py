import os
import re
import RNN_helper
import numpy as np
import pickle
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation, Dropout

from nltk import compat
from nltk.util import Index
from nltk.corpus.reader.util import *
from nltk.corpus.reader.api import *

from pickle import load
from keras.models import load_model
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import rnnpreprocessing

model = Sequential()
model.add(LSTM(150, input_shape=(len(rnnpreprocessing.X[0]), preprocessing.vocab_size)))
model.add(Dense(preprocessing.vocab_size, activation='softmax'))
model.summary
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(rnnpreprocessing.X, rnnpreprocessing.y, epochs=300, verbose=2)
model.save('RNN1.h5')
pickle.dump(mapping, open('mapping.pkl', 'wb'))

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_seq1(model, mapping, seq_length, seed_text, n_chars, temp):
    in_text = ""
    # generate a fixed number of characters
    for _ in range(n_chars):
        # encode the characters as integers
        encoded = [mapping[char] for char in seed_text]
	# truncate sequences to a fixed length
        encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
	# one hot encode
        encoded = to_categorical(encoded, num_classes=len(mapping))
        encoded = encoded.reshape(1, seq_length, 33)
	# predict character
        yhat = model.predict(encoded, verbose=0)
        new_y = sample(yhat[0], temp)
	# reverse map integer to character
        out_char = ''
        for char, index in mapping.items():
            if index == new_y:
                out_char = char
                break
	# append to input
        in_text += out_char
        seed_text += out_char
    return in_text           

model = load_model('RNN1.h5')
mapping = load(open('mapping.pkl', 'rb'))

for t in (0.60, 0.80, 1.0, 1.20):
    print('temperature = ', t)
    print(generate_seq1(model, mapping, 40, "pp shall i compare thee to a summers day ", (40*15), 1))
    print()
