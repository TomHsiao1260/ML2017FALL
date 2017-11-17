import numpy as np
from keras.models     import Sequential
from keras.layers     import Dense, Activation, Dropout, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import RMSprop, SGD, Adam
from keras.models     import load_model
from keras.utils.data_utils import get_file
import sys

test    = np.delete(np.genfromtxt(sys.argv[1] , delimiter  = ",", dtype = str),0,0)
X_test  = np.zeros((len(test),48,48))
pixel   = np.array([])
for i in range(len(test)):
    picture    = test[i][1].split()
    for j in range(48 * 48):
        pixel  = np.append(pixel, int(picture[j]))
    if np.std(pixel) != 0: std = np.std(pixel)
    else: std = 1; print('std=0')
    pixel = (pixel - np.mean(pixel)) / std
    X_test[i]  = pixel.reshape(48,48)
    pixel      = np.array([])
X_test = X_test.reshape(-1,48,48,1)
XX_test = np.repeat(X_test,3,axis=3)

#X_train = XX_train[int(len(XX_train) * 0/5):int(len(XX_train) * 4/5)]
#X_valid = XX_train[int(len(XX_train) * 4/5):int(len(XX_train) * 5/5)]
#Y_train = YY_train[int(len(XX_train) * 0/5):int(len(XX_train) * 4/5)]
#Y_valid = YY_train[int(len(XX_train) * 4/5):int(len(XX_train) * 5/5)]

weights_path = 'https://www.dropbox.com/s/j4yjitkmaz37t14/CNN_model.h5?dl=1'
path = get_file('CNN_model.h5',weights_path)
model = load_model(path)
print('model load')

#loss_val, acc_val = model.evaluate(X_valid, Y_valid)
#print()
#print('Valid_Acc : ' + str(acc_val))

file = open(sys.argv[2], 'w')
file.write("id,label\n")
for i in range(len(XX_test)):
    if i%1000==0: print(str(i)+' is finish')
    Input_x  = XX_test[i][np.newaxis,:]
    Output_y = model.predict(Input_x)[0]
    Output_y = np.argmax(Output_y)
    prediction = str(i) + "," + str(Output_y) + "\n"
    file.write(prediction)
file.close()
