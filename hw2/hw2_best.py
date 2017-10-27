import numpy as np
import sys
from keras.models      import Sequential
from keras.layers      import Dense, Activation, Dropout
from keras.optimizers  import RMSprop

def predict(input_x):
    pre_value = model.predict(input_x[np.newaxis,:])
    if pre_value[0][0] > 0.5:
        predict = 1
    else:
        predict = 0
    return predict, pre_value[0] 
    
X_train = sys.argv[3]
Y_train = sys.argv[4]
X_test  = sys.argv[5]
pre_csv = sys.argv[6]

Y_train = np.delete(np.genfromtxt(Y_train,delimiter  = ","),0,0)[:,np.newaxis]
Y_train = np.concatenate((Y_train,1-Y_train),axis=1)

X_train = np.delete(np.genfromtxt(X_train,delimiter  = ","),0,0)
X_test  = np.delete(np.genfromtxt(X_test ,delimiter  = ","),0,0)

norm    = np.mean(X_train,axis=0)  
sigma   = np.sqrt(sum((X_train-norm)**2)/len(X_train))
X_train = (X_train-norm) / sigma
X_test  = (X_test-norm)  / sigma

feature  = X_train.shape[1]

x_train  = X_train[int(len(X_train) * 0/10):int(len(X_train) * 8/10)]
y_train  = Y_train[int(len(X_train) * 0/10):int(len(X_train) * 8/10)]
x1_valid = X_train[int(len(X_train) * 8/10):int(len(X_train) * 9/10)]
x2_valid = X_train[int(len(X_train) * 9/10):int(len(X_train) * 10/10)]
y1_valid = Y_train[int(len(X_train) * 8/10):int(len(X_train) * 9/10)]
y2_valid = Y_train[int(len(X_train) * 9/10):int(len(X_train) * 10/10)]

model = Sequential()
model.add(Dense(input_dim=feature, units=100, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units=2, activation='softmax'))
rmsprop = RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer=rmsprop,
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=100, epochs=50)

file = open(pre_csv, 'w')
file.write("id,label\n")
for i in range(len(X_test)):
    Output_y = predict(X_test[i])[0]
    prediction = str(i+1) + "," + str(Output_y) + "\n"
    file.write(prediction)
file.close()





    

