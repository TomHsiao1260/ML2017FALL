import numpy as np
import sys

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
        if input_y[i] == 1 and predict(T(input_x[i]))[0] == 1:
            accuracy += 1
        elif input_y[i] == 0 and predict(T(input_x[i]))[0] == 0:
            accuracy += 1
        counts += 1
    accuracy = round(accuracy / counts, 3)
    return accuracy

def Sigmoid(input_x):
    global Weights, Biases
    z = np.dot(Weights, input_x)[0][0] + Biases
    if z > -50:
        output = 1 / (1 + np.exp(-1 * z))
    else:
        output = 0
    return output

def Gauss_parameter(input_x):
    mean    = T(np.mean(input_x,axis=0))      # feature * 1
    cov_mat = np.zeros((feature, feature))
    for i in range(len(input_x)):
        cov_mat += np.dot(T(input_x[i]) - mean, (T(input_x[i]) - mean).T)
    cov_mat = cov_mat / len(input_x)          # feature * feature
    return mean, cov_mat

def find_wb(mean_yes, mean_no, cov_share):
    sig_inv = np.linalg.inv(cov_share)
    Weights = np.dot((mean_yes-mean_no).T, sig_inv)
    Biases  = -0.5 * np.linalg.multi_dot([ mean_yes.T, sig_inv, mean_yes ])[0][0] +\
               0.5 * np.linalg.multi_dot([ mean_no .T, sig_inv, mean_no  ])[0][0] +\
               np.log(C_yes / C_no)
    return Weights, Biases

def T(input_x):
    return input_x[:,np.newaxis]

X_train = sys.argv[3]
Y_train = sys.argv[4]
X_test  = sys.argv[5]
pre_csv = sys.argv[6]

Y_train = np.delete(np.genfromtxt(Y_train,delimiter  = ","),0,0)

X_train = np.delete(np.genfromtxt(X_train,delimiter  = ","),0,0)
X_test  = np.delete(np.genfromtxt(X_test ,delimiter  = ","),0,0)

norm    = np.mean(X_train,axis=0)  
sigma   = np.sqrt(sum((X_train-norm)**2)/len(X_train))
X_train = (X_train-norm) / sigma
X_test  = (X_test-norm)  / sigma

feature  = X_train.shape[1]

train_yes = np.zeros(feature)
train_no  = np.zeros(feature)
for i in range(len(Y_train)):
    if Y_train[i] == 1:
        train_yes = np.vstack((train_yes, X_train[i]))
    else:
        train_no  = np.vstack((train_no , X_train[i]))
train_yes = np.delete(train_yes, 0, 0)   # +50k_sample * feature
train_no  = np.delete(train_no , 0, 0)   # -50k_sample * feature

C_yes = len(train_yes) / (len(train_yes)+len(train_no))
C_no  = len(train_no)  / (len(train_yes)+len(train_no))
Mean_yes, Cov_yes = Gauss_parameter(train_yes)
Mean_no , Cov_no  = Gauss_parameter(train_no)
Cov_share = C_yes * Cov_yes + C_no * Cov_no
Weights, Biases = find_wb(Mean_yes, Mean_no, Cov_share)

'''
print('Accuracy: ' + str(accuracy(X_train,Y_train)))
print()
'''

file = open(pre_csv, 'w')
file.write("id,label\n")
for i in range(16281):
    Output_y   = predict(T(X_test[i]))[0]
    prediction = str(i+1) + "," + str(Output_y) + "\n"
    file.write(prediction)
file.close()








