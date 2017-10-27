import numpy as np
import sys

def Sigmoid(input_x):
    global Weights, Biases
    z = sum(input_x * Weights + Biases)
    if z > -50:
        output = 1 / (1 + np.exp(-1 * z))
    else:
        output = 0
    return output

def SGD(input_x, input_y):
    global Weights, Biases
    dHdW = 0; dHdB = 0
    for i in range(len(input_x) // batch):
        for j in range(batch):
            index  = i * batch + j
            dHdW  += -1 * (input_y[index] - Sigmoid(input_x[index])) * input_x[index]
            dHdB  += -1 * (input_y[index] - Sigmoid(input_x[index])) * 1
        dHdW = dHdW / batch
        dHdB = dHdB / batch
        Weights -= lr_rate * dHdW + lamda * Weights
        Biases  -= lr_rate * dHdB

def predict(input_x):
    pre_value = Sigmoid(input_x)
    if pre_value > 0.5:
        predict = 1
    else:
        predict = 0
    pre_value = round(pre_value,2)
    return predict, pre_value

def accuracy(input_x, input_y):
    accuracy = 0
    counts   = 0
    for i in range(len(input_x)):
        if input_y[i] == 1 and predict(input_x[i])[0] == 1:
            accuracy += 1
        elif input_y[i] == 0 and predict(input_x[i])[0] == 0:
            accuracy += 1
        counts += 1
    accuracy = round(accuracy / counts, 3)
    return accuracy

def acc(input_x, input_y):
    accuracy = 0
    counts   = 0
    for i in range(len(input_x)):
        if input_y[i] == 1 and input_x[i][0] == 1:
            accuracy += 1
        elif input_y[i] == 0 and input_x[i][0] == 0:
            accuracy += 1
        counts += 1
    accuracy = round(accuracy / counts, 3)
    return accuracy
    
X_train = sys.argv[3]
Y_train = sys.argv[4]
X_test  = sys.argv[5]
pre_csv = sys.argv[6]

Y_train  = np.delete(np.genfromtxt(Y_train,delimiter  = ","),0,0)

XX_train = np.delete(np.genfromtxt(X_train,delimiter  = ","),0,0)[:,0:6]
XX_test  = np.delete(np.genfromtxt(X_test ,delimiter  = ","),0,0)[:,0:6]
Xx_train = np.delete(np.genfromtxt(X_train,delimiter  = ","),0,0)[:,6:74] - 0.5
Xx_test  = np.delete(np.genfromtxt(X_test ,delimiter  = ","),0,0)[:,6:74] - 0.5

norm     = np.mean(XX_train,axis=0)  
sigma    = np.sqrt(sum((XX_train-norm)**2)/len(XX_train))
XX_train = (XX_train - norm)  / sigma
XX_test  = (XX_test  - norm)  / sigma

X_train  = np.concatenate((XX_train, Xx_train), axis=1)
X_test   = np.concatenate((XX_test , Xx_test) , axis=1)

feature  = X_train.shape[1]

x_train  = X_train[0:int(len(X_train) * 8/10)]
y_train  = Y_train[0:int(len(X_train) * 8/10)]
x1_valid = X_train[int(len(X_train) * 8/10):int(len(X_train) * 9/10)]
x2_valid = X_train[int(len(X_train) * 9/10):int(len(X_train) * 10/10)]
y1_valid = Y_train[int(len(X_train) * 8/10):int(len(X_train) * 9/10)]
y2_valid = Y_train[int(len(X_train) * 9/10):int(len(X_train) * 10/10)]

lr_rate = 0.1
epoch   = 20
batch   = 100
lamda   = 0
Weights = 0.0001 * np.random.rand(feature)
Biases  = 0.0001 * np.random.rand(1)


for i in range(epoch):
    SGD(x_train, y_train)
'''
print('Acc_train  : ' + str(accuracy(x_train, y_train)))
print('Acc_valid_1: ' + str(accuracy(x1_valid, y1_valid)))
print('Acc_valid_2: ' + str(accuracy(x2_valid, y2_valid)))
print()
'''

file = open(pre_csv, 'w')
file.write("id,label\n")
for i in range(16281):
    Output_y = predict(X_test[i])[0]
    prediction = str(i+1) + "," + str(Output_y) + "\n"
    file.write(prediction)
file.close()
