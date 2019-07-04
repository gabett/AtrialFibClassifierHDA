import numpy as np
import keras
import pickle
import sys
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from sklearn.metrics import confusion_matrix
from keras.layers import Dense, Dropout, Flatten, Lambda, Reshape, GaussianNoise
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation
from keras.callbacks import ModelCheckpoint
from keras import backend as K
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

def AugGenerator(xTrain, xTest, yTrain, yTest):

    imagegen = ImageDataGenerator()

    trainGenerator = imagegen.flow(xTrain, yTrain, batch_size=20)
    testGenerator = imagegen.flow(xTest, yTest, batch_size=20)

    return trainGenerator, testGenerator

def CNN(blockSize, blockCount, inputShape, trainGen, testGen):

    model = Sequential()

    # Conv Layer
    channels = 32
    for i in range(blockCount):
        for j in range(blockSize):
            if (i, j) == (0, 0):
                conv = Conv2D(channels, kernel_size=(5, 5),
                              input_shape=inputShape, padding='same')
            else:
                conv = Conv2D(channels, kernel_size=(5, 5), padding='same')
            model.add(conv)
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(Dropout(0.15))
            if j == blockSize - 2:
                channels += 32
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        model.add(Dropout(0.15))

    # Feature aggregation across time
    model.add(Lambda(lambda x: K.mean(x, axis=1)))

    model.add(Flatten())

    # Adding noise
    model.add(GaussianNoise(0.2))

    # Linear classifier
    model.add(Dense(4, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    return model

def TrainCNN(model, trainGen, testGen, epochs):

    # Checkpoint
    filepath="./model/weights-{epoch:02d}-{val_acc:.2f}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    model.fit_generator(trainGen,
                        validation_data=testGen, steps_per_epoch = len(trainGen),
                        validation_steps = len(testGen),
                        epochs=epochs, callbacks=callbacks_list, verbose=1)

    model.save('./cnn_model.h5')

def EvaluateCNN(model, weightsFile, testGen, yTest, steps):

    model.load_weights(weightsFile)

    print('Evaluation...')
    yPredictedProbs = model.predict_generator(testGen, steps = steps)
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

        yMaxPredictedProbsForClass[~maskPred] = 0
        fpr, tpr, _ = roc_curve(yPredicted, yMaxPredictedProbsForClass)
        roc_auc = auc(fpr, tpr)

        # ROC
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic (ROC) for class ' + i)
        plt.legend(loc="lower right")
        plt.show(block=False)

        # PROC
        prec, rec, _ = precision_recall_curve(yPredicted, yMaxPredictedProbs)

        plt.figure()
        plt.plot(prec, rec, color='darkorange', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Precision')
        plt.ylabel('Recall')
        plt.title('Precision-Recall Curve (PRC) for class ' + i)
        plt.show(block=False)

    precision = sum(precisions) / 4.0
    recall = sum(recalls) / 4.0
    f1 = sum(f1Scores) / 4.0

if __name__ == '__main__':

    xTrain, yTrain = LoadTrainingSet('./TrainingSetFFT.pk1')
    xTest, yTest = LoadTrainingSet('./TestSetFFT.pk1')
    trainGen, testGen = AugGenerator(xTrain, xTest, yTrain, yTest)
    
    model = CNN(4, 6, (140, 33, 1), trainGen, testGen)
    #TrainCNN(model, trainGen, testGen, 200)
    EvaluateCNN(model, './cnn_model.h5', testGen, yTest, len(testGen))

    print('CNN completed.')