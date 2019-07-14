import numpy as np
import keras
import pickle
import sys
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from sklearn.metrics import confusion_matrix
from keras.layers import Dense, Dropout, Flatten, Lambda, Reshape, Bidirectional, LSTM, GaussianNoise
from keras.layers import Conv1D, MaxPooling1D, BatchNormalization, Activation
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
    model.add(Conv1D(8, 2, padding='same', name='conv1', input_shape = input_shape)) 
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2, name='max1'))  

    model.add(Conv1D(16, 2, padding='same', name='conv2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2, name='max2'))

    # CNN to RNN
    model.add(Reshape((1, 16)))
    
    # RNN
    model.add(Bidirectional(LSTM(200), merge_mode='ave'))
    model.add(BatchNormalization())

    model.add(Dropout(0.5))

    # Adding noise
    model.add(GaussianNoise(0.2))

    # Activation
    model.add(Dense(4, name='dense')) 
    model.add(Activation('softmax', name='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy']) 
    return model

def TrainCRNN(model, epochs):

    xTrain, yTrain = LoadTrainingSet('./TrainingSignalFeatures.pk1')
    xTest, yTest = LoadTestSet('./TestSignalFeatures.pk1')

    # Checkpoint
    filepath="./model/weights-crnn-feat-{epoch:02d}-{val_acc:.2f}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    xTrain = np.array(xTrain)
    yTrain = np.array(yTrain)

    xTest = np.array(xTest)
    yTest = np.array(yTest)

    # TODO: in produzione mettere validation_data=(xTest, yTest),
    model.fit(xTrain, yTrain, 
                        steps_per_epoch = 3328,
                        validation_steps=3328,
                        epochs=epochs, callbacks=callbacks_list, verbose=1)

    model.save('./crnn_feat_model.h5')

def EvaluateCRNN(model, weightsFile):

    xTest, yTest = LoadTestSet('./TestSetFeatureBased.pk1')

    steps = 3328
    model.load_weights(weightsFile)

    print('Evaluation...')
    yPredictedProbs = model.predict(xTest, yTest, steps = steps)
    yMaxPredictedProbs = np.amax(yPredictedProbs, axis=1)
    yPredicted = yPredictedProbs.argmax(axis = 1)
    yTest = yTest.argmax(axis=1)

    # Evaluate accuracy
    accuracy = accuracy_score(yTest, yPredicted)

    # Evaluate precision, recall and fscore
    precision, recall, fscore, _ = precision_recall_fscore_support(yTest, yPredicted, average='macro')

    precisions = []
    recalls = []
    f1Scores = []

    for i in range(4):

        yMaxPredictedProbsForClass = yMaxPredictedProbs

        # 1 * casts to int.
        maskTest = 1 * (yTest == i)
        maskPred = 1 * (yPredicted == i)
        
        precision, recall, fscore, _ = precision_recall_fscore_support(maskTest, maskPred, average='binary')

        precisions.append(precision)
        recalls.append(recall)
        f1Scores.append(fscore)

        fpr, tpr, _ = roc_curve(maskTest, yMaxPredictedProbsForClass)
        roc_auc = auc(fpr, tpr)

        # ROC
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic (ROC) for class' )
        plt.legend(loc="lower right")
        plt.savefig("./ROC_" + str(i))

        # PROC
        prec, rec, _ = precision_recall_curve(maskTest, yMaxPredictedProbsForClass)

        plt.figure()
        plt.plot(prec, rec, color='darkorange', lw=2)
        plt.xlabel('Precision')
        plt.ylabel('Recall')
        plt.title('Precision-Recall Curve (PRC) for class')
        plt.savefig("./PRC_" + str(i))

    precision = sum(precisions) / 4.0
    recall = sum(recalls) / 4.0
    f1 = sum(f1Scores) / 4.0