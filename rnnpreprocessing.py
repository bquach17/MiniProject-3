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

text = open(os.path.join(os.getcwd(), 'data/shakespeare.txt')).read()
obs = RNN_helper.parse_observations(text)

# make sequences of length 40 characters
sequences = []
for i in range(40, len(obs)):
    sequences.append(obs[i-40:i+1])
print(sequences)
def save_doc(lines, filename):
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()

# save sequences to file
out_filename = 'char_sequences.txt'
save_doc(sequences, out_filename)

def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text
 
# load
in_filename = 'char_sequences.txt'
raw_text = load_doc(in_filename)
lines = raw_text.split('\n')

chars = sorted(list(set(raw_text)))
mapping = dict((c, i) for i, c in enumerate(chars))

sq = list()
for line in lines:
    encoded_seq = [mapping[char] for char in line]
    sq.append(encoded_seq)
#33
vocab_size = len(mapping)

sq = np.array(sq)
X = sq[:,:-1]
y = sq[:,-1]

new_x = [to_categorical(x, num_classes=vocab_size) for x in X]
X = new_x
y = to_categorical(y, num_classes=vocab_size)

X = np.array(new_x)
y = np.array(y)

