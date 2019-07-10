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

def LoadTrainingSet(filename):

  with open(filename, 'rb') as f:
    xTrain, yTrain = pickle.load(f)
    return xTrain, yTrain

def LoadTestSet(filename):

  with open(filename, 'rb') as f:
    xTest, yTest = pickle.load(f)
    return xTest, yTest

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

def TrainCRNN(model, epochs):

    xTrain, yTrain = LoadTrainingSet('./TrainingSetFeatureBased.pk1')
    xTest, yTest = LoadTestSet('./TestSetFeatureBased.pk1')

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