import os
import sys
import re
import csv
import pickle
import numpy as np
import pandas as pd

from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers.merge import concatenate
from keras.models import Model, load_model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils.data_utils import get_file

MAX_NB_WORDS = 300000
MAX_SEQUENCE_LENGTH = 20
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.1

save_path = os.path.dirname(os.path.realpath(__file__))
weights_path   = 'https://www.dropbox.com/s/edeuzaq6ee3gxn8/my-text-vector.bin?dl=1'
EMBEDDING_FILE = get_file('my-text-vector.bin',weights_path, cache_subdir=save_path)
weights_path   = 'https://www.dropbox.com/s/3xs6lc9i16vhb3c/RNN.h5?dl=1'
BEST_MODLE     = get_file('RNN.h5',weights_path, cache_subdir=save_path)
SEMI_INITIAL   = 'Semi_initial.h5'
TOKENIZER      = 'tokenizer.pickle'
TRAIN_LABEL    = sys.argv[1]
TRAIN_NOLABEL  = sys.argv[2]
#TEST_FILE      = 'testing_data.txt'

num_lstm = 200
num_dense = 125
rate_drop_lstm = 0.25
rate_drop_dense = 0.25

def text_to_wordlist(text, remove_stopwords=False, stem_words=False):
    # Clean the text, with the option to remove stopwords and to stem words.    
    # Convert words to lower case and split them
    text = text.lower().split()
    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]    
    text = " ".join(text)
    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r"\s{2,}", " ", text)
    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)   
    # Return a list of words
    return(text)

########################################
## process texts in datasets
########################################
## label training datasets
file = open(TRAIN_LABEL,'r',errors='replace')
content = file.read()
line = str.split(content,'\n')
line = line[:-1]
label = []; sentence = []
for i in range(len(line)):
    split = str.split(line[i],'+++$+++')
    label.append(int(split[0]))
    sentence.append(text_to_wordlist(split[1]))
file.close()

## no label training datasets
file = open(TRAIN_NOLABEL,'r',errors='replace')
content = file.read()
line = str.split(content,'\n')
line = line[:-1]
sentence_nolabel = []
for i in range(len(line)):
    sentence_nolabel.append(text_to_wordlist(line[i]))
file.close()

'''
## testing datasets
file = open(TEST_FILE,'r',errors='replace')
content = file.read()
line = str.split(content,'\n')
line = line[1:-1]
label_test = []; sentence_test = []
for i in range(len(line)):
    split = str.split(line[i],',',1)
    label_test.append(int(split[0]))
    sentence_test.append(text_to_wordlist(split[1]))
file.close()
'''

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(sentence + sentence_nolabel)# + sentence_test)

sequences = tokenizer.texts_to_sequences(sentence)
train = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
labels = np.array(label)
word_index = tokenizer.word_index

print('Shape of train_labels tensor:', train.shape)
print('Shape of label tensor:', labels.shape)
print('Found %s unique tokens' % len(word_index))

with open(TOKENIZER, 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

########################################
## prepare embeddings
########################################
print('Preparing embedding matrix')

word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)
print('Found %s word vectors of word2vec' % len(word2vec.vocab))

nb_words = min(MAX_NB_WORDS, len(word_index))+1

embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if word in word2vec.vocab:
        embedding_matrix[i] = word2vec.word_vec(word)
print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

########################################
## sample train/validation data
########################################
perm = np.random.permutation(len(train))
idx_train = perm[:int(len(train)*(1-VALIDATION_SPLIT))]
idx_val = perm[int(len(train)*(1-VALIDATION_SPLIT)):]

data_train = train[idx_train]
labels_train = labels[idx_train]

data_val = train[idx_val]
labels_val = labels[idx_val]

weight_val = np.ones(len(labels_val))

########################################
## define the model structure
########################################
embedding_layer = Embedding(nb_words,
        EMBEDDING_DIM,
        weights=[embedding_matrix],
        input_length=MAX_SEQUENCE_LENGTH,
        trainable=False)
lstm_layer = LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
merged = lstm_layer(embedded_sequences)

merged = Dropout(rate_drop_dense)(merged)
merged = BatchNormalization()(merged)

merged = Dense(num_dense, activation='relu')(merged)
merged = Dropout(rate_drop_dense)(merged)
merged = BatchNormalization()(merged)

preds = Dense(1, activation='sigmoid')(merged)

########################################
## train the model
########################################
model = Model(inputs=[sequence_input], outputs=preds)
model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['acc'])
model.summary()
model.save(SEMI_INITIAL)

early_stopping =EarlyStopping(monitor='val_loss', patience=3)
model_checkpoint = ModelCheckpoint(BEST_MODLE, save_best_only=True, save_weights_only=False)

hist = model.fit([data_train], labels_train, \
        validation_data=([data_val], labels_val, weight_val), \
        epochs=50, batch_size=2048, shuffle=True, \
        callbacks=[early_stopping, model_checkpoint])

best_val_score = min(hist.history['val_loss'])
