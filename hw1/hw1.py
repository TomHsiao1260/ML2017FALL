import numpy as np
import csv
import sys

## 將 train.csv 或 test.csv 傳為矩陣 N*18 (累計總時數為N小時)
def sort(file_name):
#    if file_name == train_file:
#        data = np.delete(np.transpose(np.genfromtxt(train_file,delimiter = ",")),0,1)
#        delete_line = 3
    if file_name == test_file:
        data = np.transpose(np.genfromtxt(test_file,delimiter = ","))
        delete_line = 2
    split = []
    output = [] 
    for i in range(delete_line):
    	data = np.delete(data,0,0)
    for i in range(data.shape[0]):	
    	split.append(np.split(data[i],len(data[i]) / 18))
    for i in range(len(split[0])) :
        for j in range(data.shape[0]) :
            output.append(split[j][i]) 
    output = np.array(output)
    output[np.isnan(output)] = 0
    return output

## training data (for train_data)
def train(train_input,weights,bias):
    for i in range(len(train_input)-(hour_sample+1)):
        if i % (24 * 20) <= (24 * 20 - hour_sample):
            Input_x  = train_input[i:i+hour_sample]
            Output_y = np.array([[np.sum(weights * Input_x)]]) + bias
            Real_y   = np.array([[train_input[i+hour_sample][pm_position]]])
            weights, bias = SGD(weights,bias,lr_train,Input_x,Output_y,Real_y)
    return weights, bias

## training data (for test_data)
def train_test(train_input,weights,bias):
    for j in range(9-hour_sample):
        for i in range(240):
            Input_x  = train_input[9*i+j:9*i+j+hour_sample]
            Output_y = np.array([[np.sum(weights * Input_x)]]) + bias
            Real_y   = np.array([[train_input[9*i+j+hour_sample][pm_position]]])
            weights, bias = SGD(weights,bias,lr_test,Input_x,Output_y,Real_y)       
    return weights, bias

## Stochastic Gradient Descent
def SGD(weights,bias,l_rate,Input_xx,Output_yy,Real_yy):
    dL_bias    = 2 * (Real_yy - Output_yy) * (-1)
    dL_weights = 2 * (Real_yy - Output_yy) * (-1) * Input_xx
    bias       = bias    - l_rate * dL_bias
    weights    = weights - l_rate * dL_weights - 2 * lambda_rate * weights
    return weights, bias

## error calculator (for train_data)
def error(data_input):
    counts = 0; error  = 0
    for i in range(len(data_input)-(hour_sample+1)):
        Input_x  = data_input[i:i+hour_sample]  / norm_factor
        Output_y = Predict(Input_x)
        Real_y   = data_input[i+hour_sample][pm_position]
        counts  += 1
        error   += (Real_y-Output_y)**2
    if counts == 0:
        error = np.nan
    else:
        error = np.sqrt(error/counts)
    return error

## error calculator (for test_data)
def error_test(data_input):
    counts = 0; error  = 0
    for j in range(9-hour_sample):
        for i in range(240):
            Input_x  = data_input[9*i+j:9*i+j+hour_sample] / norm_factor
            Output_y = Predict(Input_x)
            Real_y   = data_input[9*i+j+hour_sample][pm_position]
            counts  += 1
            error   += (Real_y - Output_y)**2
    if counts == 0:
        error = np.nan
    else:
        error = np.sqrt(error/counts)
    return error

## Predict
def Predict(input_x):
    predict =\
        (np.sum(input_x * Weights) + Bias[0][0]) * norm_factor[pm_position]
    return predict

## Initialize
test_csv   = sys.argv[1]
result_csv = sys.argv[2]

iteration      = 500
iteration_test = 700
lr_train       = 0.00001
lr_test        = 0.000001
lambda_rate    = 0.0000001
hour_sample    = 8
Para_degree    = 2

parameter_18   = np.array([0,0,0,0,0,0,0,1,0,1,0,0,1,0,0,0,0,0])
para_numbers   = np.count_nonzero(parameter_18)
pm_position    = np.count_nonzero(parameter_18[:10])-1

Weights = 0.01 * np.random.rand(hour_sample , para_numbers* Para_degree)
Bias    = 0.01 * np.random.rand(1,1)

## Import data
#train_file = 'train.csv'
test_file  = test_csv
#train_data = sort(train_file)
test_data  = sort(test_file)
column_delete = 0 
for i in range(18):
    if parameter_18[i] == 0:
        #train_data = np.delete(train_data, i - column_delete, axis=1)
        test_data  = np.delete(test_data, i - column_delete, axis=1)
        column_delete += 1
if Para_degree == 2:
    #train_data = np.concatenate((train_data,train_data**2),axis=1)
    test_data  = np.concatenate((test_data,test_data**2),axis=1)  
        
## normalization & flatten data (所有和W,b有關的運算都要先norm後代入才會對)
#norm_factor   = np.mean(train_data,axis=0)
#train_set     = train_data[0:int(len(train_data) * 9/10)]
#validate_set  = train_data[int(len(train_data) * 9/10):len(train_data)]
#norm_trainset = train_set / norm_factor
norm_factor = np.load('norm_factor.npy')
norm_testdata = test_data / norm_factor
     
'''
## train_data training
for i in range(iteration):
    Weights, Bias = train(norm_trainset,Weights,Bias)

## test_data training
for i in range(iteration_test):
    Weights, Bias = train_test(norm_testdata,Weights,Bias)

np.save('linear_w.npy',Weights)
np.save('linear_b.npy',Bias)
np.save('norm_factor.npy',norm_factor)
'''

Weights = np.load('linear_w.npy')
Bias    = np.load('linear_b.npy')
        
## test_data prediction
file = open(result_csv, 'w')
file.write("id,value\n")
for i in range(240):
    Output_y = Predict(norm_testdata[9*(i+1)-hour_sample:9*(i+1)])
    prediction = "id_" + str(i) + "," + str(float(Output_y)) + "\n"
    file.write(prediction)
file.close()

        





