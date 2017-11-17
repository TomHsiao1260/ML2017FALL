import numpy as np
from keras.models     import Sequential
from keras.layers     import Dense, Activation, Dropout, Convolution2D, MaxPooling2D, Flatten, LeakyReLU
from keras.optimizers import RMSprop, SGD, Adam
from keras.models     import load_model
from keras import regularizers

model = load_model('CNN_model.h5')

#YY_train = np.load('Y_train.npy')
#XX_train = np.load('X_train.npy')
#YY_train_flipamp1 = np.load('Y_train_flipamp1.npy')
#XX_train_flipamp1 = np.load('X_train_flipamp1.npy')
#YY_train_flipamp2 = np.load('Y_train_flipamp2.npy')
#XX_train_flipamp2 = np.load('X_train_flipamp2.npy')
#YY_train_flipamp3 = np.load('Y_train_flipamp3.npy')
#XX_train_flipamp3 = np.load('X_train_flipamp3.npy')
#YY_train_flipamp4 = np.load('Y_train_flipamp4.npy')
#XX_train_flipamp4 = np.load('X_train_flipamp4.npy')

#X_train = XX_train[int(len(XX_train) * 0/5):int(len(XX_train) * 4/5)]
#X_valid = XX_train[int(len(XX_train) * 4/5):int(len(XX_train) * 5/5)]
#Y_train = YY_train[int(len(XX_train) * 0/5):int(len(XX_train) * 4/5)]
#Y_valid = YY_train[int(len(XX_train) * 4/5):int(len(XX_train) * 5/5)]

#len_X = len(XX_train_flipamp1)
#X_train_flipamp1 = XX_train_flipamp1[int(len_X * 0/5):int(len_X * 1/5)]
#Y_train_flipamp1 = YY_train_flipamp1[int(len_X * 0/5):int(len_X * 1/5)]
#X_train_flipamp2 = XX_train_flipamp2[int(len_X * 0/5):int(len_X * 1/5)]
#Y_train_flipamp2 = YY_train_flipamp2[int(len_X * 0/5):int(len_X * 1/5)]
#X_train_flipamp3 = XX_train_flipamp3[int(len_X * 0/5):int(len_X * 1/5)]
#Y_train_flipamp3 = YY_train_flipamp3[int(len_X * 0/5):int(len_X * 1/5)]
#X_train_flipamp4 = XX_train_flipamp4[int(len_X * 0/5):int(len_X * 1/5)]
#Y_train_flipamp4 = YY_train_flipamp4[int(len_X * 0/5):int(len_X * 1/5)]

'''
# Block 1
model = Sequential()
model.add(Convolution2D(64, 3, strides=1, padding='same',
          batch_input_shape=(None,48,48,3), name='block1_conv1'))
model.add(LeakyReLU(alpha=0.05))
model.add(Convolution2D(64, 3, strides=1, padding='same', name='block1_conv2'))
model.add(LeakyReLU(alpha=0.05))
model.add(MaxPooling2D(2, strides=2, padding='same', name='block1_pool'))
model.add(Dropout(0.1))

# Block 2
model.add(Convolution2D(128, 3, strides=1, padding='same', name='block2_conv1'))
model.add(LeakyReLU(alpha=0.05))
model.add(Convolution2D(128, 3, strides=1, padding='same', name='block2_conv2'))
model.add(LeakyReLU(alpha=0.05))
model.add(MaxPooling2D(2, strides=2, padding='same', name='block2_pool'))
model.add(Dropout(0.1))

# Block 3
model.add(Convolution2D(256, 3, strides=1, padding='same', name='block3_conv1'))
model.add(LeakyReLU(alpha=0.05))
model.add(Convolution2D(256, 3, strides=1, padding='same', name='block3_conv2'))
model.add(LeakyReLU(alpha=0.05))
model.add(MaxPooling2D(2, strides=2, padding='same', name='block3_pool'))
model.add(Dropout(0.4))

# FC flatten
model.add(Flatten())
model.add(Dense(2048, kernel_regularizer=regularizers.l2(0.01),name='fc'))
model.add(LeakyReLU(alpha=0.05))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax', name='predictions'))
'''
'''
model.summary()

# Optimizer + compile
adam = Adam(lr=1e-6)
model.compile(optimizer = adam, loss = 'categorical_crossentropy', metrics = ['accuracy'])

# training + evaluate
epoch = 20
for i in range(epoch):
    print('epochs : ' + str(i + 1))
    if i % 4 == 0: model.fit(X_train_flipamp1, Y_train_flipamp1, epochs=1, batch_size=400)
    if i % 4 == 1: model.fit(X_train_flipamp2, Y_train_flipamp2, epochs=1, batch_size=400)
    if i % 4 == 2: model.fit(X_train_flipamp3, Y_train_flipamp3, epochs=1, batch_size=400)
    if i % 4 == 3: model.fit(X_train_flipamp4, Y_train_flipamp4, epochs=1, batch_size=400)
    loss_val, acc_val = model.evaluate(X_valid, Y_valid)
    print()
    print('Valid_Acc : ' + str(acc_val))

    if i == 0: model.save('CNN_model(1).h5')
    if i == 1: model.save('CNN_model(2).h5')
    if i == 2: model.save('CNN_model(3).h5')
    if i == 3: model.save('CNN_model(4).h5')
    if i == 4: model.save('CNN_model(5).h5')
    if i == 5: model.save('CNN_model(6).h5')
    if i == 6: model.save('CNN_model(7).h5')
    if i == 7: model.save('CNN_model(8).h5')
    if i == 8: model.save('CNN_model(9).h5')
    if i == 9: model.save('CNN_model(10).h5')
    if i == 10: model.save('CNN_model(11).h5')
    if i == 11: model.save('CNN_model(12).h5')
    if i == 12: model.save('CNN_model(13).h5')
    if i == 13: model.save('CNN_model(14).h5')
    if i == 14: model.save('CNN_model(15).h5')
    if i == 15: model.save('CNN_model(16).h5')
    if i == 16: model.save('CNN_model(17).h5')
    if i == 17: model.save('CNN_model(18).h5')
    if i == 18: model.save('CNN_model(19).h5')
    if i == 19: model.save('CNN_model(20).h5')
'''
