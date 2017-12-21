import sys
import pickle
import numpy as np
from numpy import random

from keras.layers import Dense, Input, Embedding, Dropout, Activation, Flatten, Concatenate
from keras.models import Model, load_model

TEST    = sys.argv[1]
PREDICT = sys.argv[2]
MOVIE   = sys.argv[3]
USER    = sys.argv[4]

mean = 3.58124186
std  = 1.11656855

MODEL      = './model/'
FACTOR     = MODEL  + 'factor/'
user_pkl   = FACTOR + 'user.pkl'
movie_pkl  = FACTOR + 'movie.pkl'
u_pkl      = FACTOR + 'u_detail.pkl'
m_pkl      = FACTOR + 'm_detail.pkl'
user       = pickle.load(open(user_pkl,'rb'))
movie      = pickle.load(open(movie_pkl,'rb'))
age_gender = pickle.load(open(u_pkl,'rb'))
genre      = pickle.load(open(m_pkl,'rb'))
BEST_MODLE = MODEL + 'model_DNN.h5'

test       = np.delete(np.genfromtxt( TEST, delimiter  = ",", dtype = str),0,0)
user_test  = np.zeros((len(test),1))
movie_test = np.zeros((len(test),1))
u_test     = np.zeros((len(test),7))
m_test     = np.zeros((len(test),18))
for i in range(len(test)):
    user_test[i][0]  = user[test[i][1]]
    movie_test[i][0] = movie[test[i][2]]
    u_test[i] = np.asarray(age_gender[test[i][1]])
    m_test[i] = np.asarray(genre[test[i][2]])

model = load_model(BEST_MODLE)
preds = model.predict([user_test, movie_test, u_test, m_test], batch_size=1000, verbose=1)

with open(PREDICT, 'w') as f:
    f.write('TestDataID,Rating\n')
    for i in range(len(test)):
        predict = (preds[i][0] * std) + mean
        if predict > 5: predict = 5
        if predict < 1: predict = 1
        f.write(str(i+1)+','+str(predict)+'\n')
