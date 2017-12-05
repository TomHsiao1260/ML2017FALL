import os
import sys
import re
import csv
import pickle
import numpy as np
import pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.models import Model, load_model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils.data_utils import get_file

MAX_NB_WORDS = 100000
MAX_SEQUENCE_LENGTH = 20
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.1

save_path = os.path.dirname(os.path.realpath(__file__))
weights_path = 'https://www.dropbox.com/s/3xs6lc9i16vhb3c/RNN.h5?dl=1'
BEST_MODLE = get_file('RNN.h5', weights_path, cache_subdir=save_path)
TOKENIZER  = 'tokenizer.pickle'
TEST_FILE  = sys.argv[1]
PREDICT    = sys.argv[2]

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
## process texts in datasets (test data)
########################################
file = open(TEST_FILE,'r',errors='replace')
content = file.read()
line = str.split(content,'\n')
line = line[1:-1]
idx_test = []; sentence_test = []
for i in range(len(line)):
    split = str.split(line[i],',',1)
    idx_test.append(int(split[0]))
    sentence_test.append(text_to_wordlist(split[1]))
file.close()

with open(TOKENIZER, 'rb') as handle:
    tokenizer = pickle.load(handle)

sequences_test = tokenizer.texts_to_sequences(sentence_test)
data_test = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH)
id_test = np.array(idx_test)

########################################
## make the submission
########################################
print('Start making the submission')

model = load_model(BEST_MODLE)

preds = model.predict([data_test], batch_size=8192, verbose=1)
preds = np.round(preds).astype(int)

submission = pd.DataFrame({'id':id_test, 'label':preds.ravel()})
submission.to_csv(PREDICT, index=False)
