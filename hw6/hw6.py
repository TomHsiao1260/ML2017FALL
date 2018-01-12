import sys
import numpy as np
import pickle

MODEL   = './model/'
TEST    = sys.argv[2]
PREDICT = sys.argv[3]
CLASS_1 = MODEL + 'class_1.pkl'
CLASS_2 = MODEL + 'class_2.pkl'
CLASS_3 = MODEL + 'class_3.pkl'
class_1 = pickle.load(open(CLASS_1,'rb'))
class_2 = pickle.load(open(CLASS_2,'rb'))
class_3 = pickle.load(open(CLASS_3,'rb'))

test = np.delete(np.genfromtxt(TEST, delimiter  = ",", dtype = str),0,0)

with open(PREDICT, 'w') as f:
    f.write('ID,Ans\n')
    for i in range(len(test)):
        if class_1[int(test[i][1])]==class_2[int(test[i][1])]==class_3[int(test[i][1])]==1:
            if class_1[int(test[i][2])]==class_2[int(test[i][2])]==class_3[int(test[i][2])]==1:
                pred = 1
            if class_1[int(test[i][2])]==class_2[int(test[i][2])]==class_3[int(test[i][2])]==-1:
                pred = 0
        elif class_1[int(test[i][1])]==class_2[int(test[i][1])]==class_3[int(test[i][1])]==-1:
            if class_1[int(test[i][2])]==class_2[int(test[i][2])]==class_3[int(test[i][2])]==-1:
                pred = 1
            if class_1[int(test[i][2])]==class_2[int(test[i][2])]==class_3[int(test[i][2])]==1:
                pred = 0
        else :
            pred = 0
        f.write(str(i)+','+str(pred)+'\n')

