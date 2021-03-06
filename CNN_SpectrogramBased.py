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
        
        yTrain[yTrain=='N'] = 0
        yTrain[yTrain=='A'] = 1
        yTrain[yTrain=='O'] = 2
        yTrain[yTrain=='~'] = 3

        # Count the elements in the sets
        num_train_data_normal = sum(yTrain == 0)
        num_train_data_afib   = sum(yTrain == 1)
        num_train_data_other = sum(yTrain == 2)
        num_train_data_noise   = sum(yTrain == 3)

        print('### TRAIN SET')
        print('\tNormal ECG: {} ({:.2f}%)'.format(num_train_data_normal, 100 * num_train_data_normal / len(yTrain)))
        print('\tAfib ECG: {} ({:.2f}%)'.format(num_train_data_afib, 100 * num_train_data_afib / len(yTrain)))
        print('\tOther ECG: {} ({:.2f}%)'.format(num_train_data_other, 100 * num_train_data_other / len(yTrain)))
        print('\tNoisy ECG: {} ({:.2f}%)'.format(num_train_data_noise, 100 * num_train_data_noise / len(yTrain)))
        

        yTrain = keras.utils.to_categorical(yTrain)

        return xTrain, yTrain

def LoadTestSet(filename):

    with open(filename, 'rb') as f:
        xTest, yTest = pickle.load(f)

        yTest[yTest=='N'] = 0
        yTest[yTest=='A'] = 1
        yTest[yTest=='O'] = 2
        yTest[yTest=='~'] = 3

        num_val_data_normal   = sum(yTest == 0)
        num_val_data_afib     = sum(yTest == 1)
        num_val_data_other = sum(yTest == 2)
        num_val_data_noise   = sum(yTest == 3)

        print('### VALIDATION SET')
        print('\tNormal ECG: {} ({:.2f}%)'.format(num_val_data_normal, 100 * num_val_data_normal / len(yTest)))
        print('\tAfib ECG: {} ({:.2f}%)'.format(num_val_data_afib, 100 * num_val_data_afib / len(yTest)))
        print('\tOther ECG: {} ({:.2f}%)'.format(num_val_data_other, 100 * num_val_data_other / len(yTest)))
        print('\tNoisy ECG: {} ({:.2f}%)'.format(num_val_data_noise, 100 * num_val_data_noise / len(yTest)))

        yTest = keras.utils.to_categorical(yTest)

        return xTest, yTest

def AugGenerator(xTrain, xTest, yTrain, yTest):

    imagegen = ImageDataGenerator()

    trainGenerator = imagegen.flow(xTrain, yTrain, batch_size=32)
    testGenerator = imagegen.flow(xTest, yTest, batch_size=32)

    return trainGenerator, testGenerator

def CNN(blockSize, blockCount, inputShape):

    model = Sequential()

    channels = 32
    for i in range(blockCount):

        for j in range(blockSize):

            if i == 0 and j == 0:

                conv = Conv2D(channels, kernel_size=(5, 5),
                              input_shape=inputShape, padding='same')
                              
            else:
                conv = Conv2D(channels, kernel_size=(5, 5), padding='same')

            model.add(conv)
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(Dropout(0.3))

            if j == blockSize - 2:
                channels += 32

        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        model.add(Dropout(0.3))

    model.add(Flatten())

    # Adding noise
    model.add(GaussianNoise(0.2))

    # Linear classifier
    model.add(Dense(4, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    return model

def TrainCNN(model, epochs):

    xTrain, yTrain = LoadTrainingSet('./TrainingSetFFT.pk1')
    xTest, yTest = LoadTrainingSet('./TestSetFFT.pk1')
    
    trainGen, testGen = AugGenerator(xTrain, xTest, yTrain, yTest)

    # Checkpoint
    filepath = "./model/weights-cnn-{epoch:02d}-{val_acc:.2f}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    model.fit_generator(trainGen,
                        validation_data=testGen, steps_per_epoch = len(trainGen) / 32,
                        validation_steps = len(testGen) / 32,
                        epochs=epochs, callbacks=callbacks_list, verbose=1)

    model.save('./cnn_model.h5')

def EvaluateCNN(model, weightsFile):

    xTest, yTest = LoadTestSet('./TestSetFFT.pk1')
    
    model.load_weights(weightsFile)

    print('Evaluation...')
    yPredictedProbs = model.predict(xTest)
    yMaxPredictedProbs = np.amax(yPredictedProbs, axis=1)
    yPredicted = yPredictedProbs.argmax(axis = 1)
    yTest = yTest.argmax(axis=1)

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

