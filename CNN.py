import numpy as np
import keras
import pickle
import sys
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from sklearn.metrics import confusion_matrix
from keras.layers import Dense, Dropout, Flatten, Lambda, Reshape
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation
from keras import backend as K


def LoadTrainingSet(filename):

    with open(filename, 'rb') as f:
        df = pickle.load(f)
        return df

def LoadTestSet(filename):

    with open(filename, 'rb') as f:
        df = pickle.load(f)
        return df
    

def AugGenerator(trainingSet, testSet):

    imagegen = ImageDataGenerator()
    trainGen = imagegen.flow_from_dataframe(dataframe=trainingSet, x_col="signal", y_col="label", class_mode="categorical", batch_size=20)
    testGen = imagegen.flow_from_dataframe(dataframe=testSet, x_col="signal", y_col="label", class_mode="categorical", batch_size=20)
    
    return trainGen, testGen


def CNN(blockSize, blockCount, inputShape, trainGen, testGen, epochs):

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

    # Linear classifier
    model.add(Dense(4, activation='softmax'))


    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy']) # F1?


    model.fit_generator(trainGen,
                        validation_data=testGen, steps_per_epoch = trainGen.x.size // 20,
                        validation_steps = testGen.x.size // 20,
                        epochs=epochs, verbose=1)
    return model


if __name__ == '__main__':

    trainingSet = LoadTrainingSet('./TrainingSet.pk1')
    testSet = LoadTrainingSet('./TestSet.pk1')
    trainGen, testGen = AugGenerator(trainingSet, testSet)
    
    model = CNN(4, 6, (285, 33, 1), trainGen, testGen, 1)
    model.save('./cnn_model.h5')
    model = keras.models.load_model('./cnn_model.h5')
    model.summary()

    print('Evaluation...')
    y_predict = model.predict_generator(testGen, steps = testGen.x.size // 20).argmax(axis=1)
    print(y_predict)
    yTest = yTest.argmax(axis=1)

    f = open('./evaluation_CNN.txt', 'w')
    f.write('model: CNN, epochs: {} \n confusion_matrix: \n {}'.format(1, confusion_matrix(yTest, y_predict)))
    f.close()    

    print('CNN completed.')