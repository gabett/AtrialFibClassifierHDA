import numpy as np
import keras
import pickle
import sys
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from sklearn.metrics import confusion_matrix
from keras.layers import Dense, Dropout, Flatten, Lambda, Reshape, Bidirectional, LSTM, GaussianNoise, GRU
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

def generator(x, y, batchSize = 32):

  offset = 0

  while True:
    
    if(offset > len(x)):
      offset = 0

    xBatch = []
    yBatch = []

    rightLim = offset + batchSize
    xSamples = x[offset : rightLim]
    ySamples = y[offset : rightLim]

    for i, xSamp in enumerate(xSamples):
      xTemp = xSamp
      yTemp = ySamples[i]
      #x = np.expand_dims(x, axis=-1)
      xBatch.append(xTemp)
      yBatch.append(yTemp)

    offset += batchSize - 1
    xBatch = np.array(xBatch)
    
    yBatch = np.array(yBatch)

    yield xBatch, yBatch

def RNN(input_shape):
  
    X_input = Input(input_shape)

    X = GRU(128, name='gru0')(X_input)

    X = Dropout(0.5)(X)
    X = GaussianNoise(0.2)(X)
    X = Dense(4, activation='softmax', name='fc')(X)

    model = Model(inputs = X_input, outputs = X, name='RNN')

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy']) 
        
    print(model.summary())

    return model

def TrainRNN(model, epochs):

    xTrain, yTrain = LoadTrainingSet('./TrainingSignalFeatures.pk1')
    xTest, yTest = LoadTestSet('./TestSignalFeatures.pk1')

    # Checkpoint
    filepath="./model/weights-rnn-feat-{epoch:02d}-{val_acc:.2f}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    trainGen = generator(xTrain, yTrain)
    testGen = generator(xTest, yTest)

    model.fit_generator(trainGen, validation_data=testGen, 
                        steps_per_epoch = np.ceil(len(xTrain) / 32),
                        validation_steps=np.ceil(len(xTest) / 32),
                        epochs=epochs, callbacks=callbacks_list, verbose=1)

    model.save('./rnn_feat_model.h5')

def EvaluateRNN(model, weightsFile):

    xTest, yTest = LoadTestSet('./TestSignalFeatures.pk1')

    model.load_weights(weightsFile)

    xTest = np.array(xTest)
    yTest = np.array(yTest)

    print('Evaluation...')
    yPredictedProbs = model.predict(xTest)
    yMaxPredictedProbs = np.amax(yPredictedProbs, axis=1)
    yPredicted = yPredictedProbs.argmax(axis = 1)
    yTest = yTest.argmax(axis=1)

    # Evaluate precision, recall and fscore
    precision, recall, fscore, _ = precision_recall_fscore_support(yTest, yPredicted, average='macro')

    print('Precision: ', str(precision))
    print('Recall: ', str(recall))
    print('F-Score: ', str(fscore))

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

        print('Class ' , str(i))
        print('Precision: ', str(precision))
        print('Recall: ', str(recall))
        print('F-Score: ', str(fscore))

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

    print('Overall precision: ', str(precision))
    print('Overall recall: ', str(recall))
    print('Overall F-Score: ', str(f1))
