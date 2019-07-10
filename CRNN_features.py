import numpy as np
import keras
import pickle
import sys
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from sklearn.metrics import confusion_matrix
from keras.layers import Dense, Dropout, Flatten, Lambda, Reshape, Bidirectional, LSTM, GaussianNoise
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras import Input
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc, accuracy_score, precision_recall_curve
import matplotlib.pyplot as plt 


def DataGenerator(qAmplitudes, rAmplitudes, heartRatesDuringHeartBeat, rrIntervals, qrsDurations, labels, batch_size, trainSplit):

    batch = []
    inputs = []

    while True:
        for i in range(batch_size):
          inputs = np.vstack((rAmplitudes[i, :], qAmplitudes[i, :]))
          inputs = np.vstack((inputs, heartRatesDuringHeartBeat[i, :]))
          inputs = np.vstack((inputs, rrIntervals[i, :]))
        
          batch.append(inputs.reshape(4,13))
         
          inputs = []
      
        trainDim = batch_size - int(batch_size * trainSplit)
        xTrain = np.array(batch[:trainDim])
        xTest = np.array(batch[trainDim:])

        y = labels[:batch_size]
        y = keras.utils.to_categorical(y)

        yTrain = y[:trainDim]
        yTest = y[trainDim:]

        return xTrain, yTrain, xTest, yTest
        

def CRNN(input_shape):
  
    model = Sequential()
    # CNN
    model.add(Conv2D(8, (2, 2), padding='same', name='conv1', input_shape = input_shape)) 
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='max1'))  

    model.add(Conv2D(16, (2, 2), padding='same', name='conv2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='max2'))

    # CNN to RNN
    model.add(Reshape((3, 16)))
    
    # RNN
    model.add(Bidirectional(LSTM(200), merge_mode='ave'))
    model.add(BatchNormalization())

    model.add(Dropout(0.5))

    # Adding noise
    model.add(GaussianNoise(0.2))

    # Activation
    model.add(Dense(3, name='dense')) 
    model.add(Activation('softmax', name='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy']) 
    return model

def TrainCRNN(model, xTrain, yTrain, xTest, yTest, epochs):

    # Checkpoint
    filepath="./model/weights-crnn-feat-{epoch:02d}-{val_acc:.2f}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    xTrain = np.expand_dims(xTrain, axis = -1)
    xTest = np.expand_dims(xTest, axis = -1)
    model.fit(xTrain, yTrain, validation_data=(xTest, yTest),
                        steps_per_epoch = 3328,
                        validation_steps=3328,
                        epochs=epochs, callbacks=callbacks_list, verbose=1)

    model.save('./crnn_feat_model.h5')