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
def train(input_x, L1_weights, L1_biases, L2_weights, L2_biases):
    for i in range(len(input_x)-(hour_sample+1)):
        if i % (24 * 20) <= (24 * 20 - hour_sample):
            L1_output, L2_output =\
                NN(flat(input_x[i:i+hour_sample]), L1_weights, L1_biases, L2_weights, L2_biases)
            real_y = np.array([[input_x[i+hour_sample][pm_position]]])
            L1_weights, L1_biases, L2_weights, L2_biases =\
                GradientDescent(flat(input_x[i:i+hour_sample]), L1_weights, L1_biases, L2_weights, L2_biases,
                lr_train, L1_output, L2_output, real_y)
    return L1_weights, L1_biases, L2_weights, L2_biases

## training data (for test_data)
def train_test(input_x, L1_weights, L1_biases, L2_weights, L2_biases):
    input_x = np.append(input_x[0:1], input_x, axis=0)
    for i in range(2):
        L1_output, L2_output =\
            NN(flat(input_x[i:i+hour_sample]), L1_weights, L1_biases, L2_weights, L2_biases)
        real_y = np.array([[input_x[i+hour_sample][pm_position]]])
        L1_weights, L1_biases, L2_weights, L2_biases =\
            GradientDescent(flat(input_x[i:i+hour_sample]), L1_weights, L1_biases, L2_weights, L2_biases,
            lr_test, L1_output, L2_output, real_y)
    return L1_weights, L1_biases, L2_weights, L2_biases

## GradientDescent
def GradientDescent(input_x, L1_weights, L1_bias, L2_weights, L2_bias, l_rate,
        L1_output, L2_output, real_y):
    L1_dCdW, L1_dCdB, L2_dCdW, L2_dCdB  = Back_propagation(input_x, L1_output, L2_output, real_y, L2_weights)
    L1_weights = L1_weights - l_rate * L1_dCdW - 2 * L1_weights * lambda_rate 
    L1_bias    = L1_bias    - l_rate * L1_dCdB                                
    L2_weights = L2_weights - l_rate * L2_dCdW - 2 * L2_weights * lambda_rate 
    L2_bias    = L2_bias    - l_rate * L2_dCdB                                
    return L1_weights, L1_bias, L2_weights, L2_bias

## Back_propagation
def Back_propagation(input_x, L1_output, L2_output, real_y, L2_weights):
    # weights term (forward pass)
    L1_dzdw = input_x
    L2_dzdw = L1_output
    # weights term (backward pass)
    L2_dcdy = 2 * (real_y - L2_output) * (-1)
    L2_dydz = 1
    L2_dcdz = L2_dcdy * L2_dydz
    L2_dcdw = np.dot(L2_dcdy, L2_dzdw.T)
    L1_dcdx = np.dot(L2_weights.T, L2_dcdy)
    L1_dxdz = Diff_ReLu(Inv_ReLu(L1_output))
    L1_dcdz = L1_dcdx * L1_dxdz
    L1_dcdw = np.dot(L1_dcdz, L1_dzdw.T)
    # bias term
    L2_dcdb = 1 * L2_dcdz
    L1_dcdb = 1 * L1_dcdz
    return L1_dcdw, L1_dcdb, L2_dcdw, L2_dcdb

# NN function
def NN(input_x, L1_weights, L1_biases, L2_weights, L2_biases):
    L1_output = NN_layer(input_x, L1_weights, L1_biases, 'ReLu')
    L2_output = NN_layer(L1_output, L2_weights, L2_biases, None)
    return L1_output, L2_output
def NN_layer(input_x, weights, biases, activation_function=None):
    Wx_plus_b = np.dot(weights, input_x) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = ReLu(Wx_plus_b)
    return outputs

# ReLu related function
def ReLu(input_x):
    output = (abs(input_x) + input_x) / 2          
    return output
def Inv_ReLu(input_x):
    output = input_x
    return output
def Diff_ReLu(input_x):
    output = np.heaviside(abs(input_x) + input_x, 0)
    return output

## error calculator (for train_data)
def error(data_input):
    counts = 0; error  = 0
    for i in range(len(data_input)-(hour_sample+1)):
        L1_Output, L2_Output =\
            NN(flat(data_input[i:i+hour_sample]/norm_factor), L1_W, L1_B, L2_W, L2_B)
        Output_y = L2_Output[0][0] * norm_factor[pm_position]
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
            L1_Output, L2_Output =\
                NN(flat(data_input[9*i+j:9*i+j+hour_sample]/norm_factor), L1_W, L1_B, L2_W, L2_B)
            Output_y = L2_Output[0][0] * norm_factor[pm_position]
            Real_y   = data_input[9*i+j+hour_sample][pm_position]
            counts  += 1
            error   += (Real_y-Output_y)**2
    if counts == 0:
        error = np.nan
    else:
        error = np.sqrt(error/counts)
    return error

## flatten inputs  (N, parameter) to (N * parameter, 1) 
def flat(data_input):
    data_flatten = data_input.flatten()[:,np.newaxis]
    return data_flatten

## Initialize
test_csv   = sys.argv[1]
result_csv = sys.argv[2]

iteration      = 700
iteration_test = 500
lr_train       = 0.00001
lr_test        = 0.000001
#lambda_rate    = 0.0000001
lambda_rate    = 0
hour_sample    = 8
Para_degree    = 2

parameter_18   = np.array([0,0,0,0,0,0,0,1,0,1,0,0,1,0,0,0,0,0])
para_numbers   = np.count_nonzero(parameter_18)
pm_position    = np.count_nonzero(parameter_18[:10])-1

in_size     = hour_sample * para_numbers * Para_degree
hidden_size = 5
out_size    = 1

L1_W = 0.01 * np.random.rand(hidden_size,in_size)
L1_B = 0.01 * np.random.rand(hidden_size,1)
L2_W = 0.01 * np.random.rand(out_size,hidden_size)
L2_B = 0.01 * np.random.rand(out_size,1)

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
#norm_trainset = train_set    / norm_factor
#norm_validset = validate_set / norm_factor
norm_factor = np.load('norm_factor.npy')
norm_testdata = test_data    / norm_factor

'''
## train_data training
for i in range(iteration):
    L1_W, L1_B, L2_W, L2_B = train(norm_trainset, L1_W, L1_B, L2_W, L2_B)

np.save('NN_L1_w.npy',L1_W)
np.save('NN_L1_b.npy',L1_B)
np.save('NN_L2_w.npy',L2_W)
np.save('NN_L2_b.npy',L2_B)
'''

L1_W = np.load('NN_L1_w.npy')
L1_B = np.load('NN_L1_b.npy')
L2_W = np.load('NN_L2_w.npy')
L2_B = np.load('NN_L2_b.npy')

## test_data training
predict = [None] * 240
for i in range(100,iteration_test+1,100):
    for j in range(240):
        L1_w, L1_b, L2_w, L2_b = L1_W, L1_B, L2_W, L2_B
        for k in range(i):
            L1_w, L1_b, L2_w, L2_b =\
                train_test(norm_testdata[9*j:9*j+1+hour_sample], L1_w, L1_b, L2_w, L2_b)
        L1_Output, L2_Output =\
            NN(flat(norm_testdata[9*j+1:9*j+1+hour_sample]), L1_w, L1_b, L2_w, L2_b)
        predict[j] = L2_Output[0][0] * norm_factor[pm_position]

## test_data prediction
file = open(result_csv, 'w')
file.write("id,value\n")
for i in range(240):
    prediction = "id_" + str(i) + "," + str(float(predict[i])) + "\n"
    file.write(prediction)
file.close()   


        





