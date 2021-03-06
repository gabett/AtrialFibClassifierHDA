import numpy as np
import pandas as pd
import sys
import wfdb
from wfdb import processing 
from scipy.signal import medfilt
import scipy
import keras
import pickle
import os.path
import CRNN_FeatureBased
import QRS_util
from tqdm import tqdm
from random import seed
from sklearn.model_selection import train_test_split

folderPath = './training2017/'
recordsPath = folderPath + 'REFERENCE.csv'

def TrainTestSplit(signals, labels):

    if len(signals) and len(labels)> 0:
        df = {

            'signal' : signals,
            'label' : labels
        }

        df = pd.DataFrame(df, columns = ['signal', 'label'])

        # Keep 20% of the data out for validation
        train_reference_df, val_reference_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=123)

        # 'N' = 0
        # 'A' = 1
        # 'O' = 2
        # '~' = 3

        return train_reference_df, val_reference_df

def CreateNoiseVector():
    
    noise = np.random.normal(0, 2000, size = (9000,))
    noise = noise.astype(np.int64)
    return noise

def GenerateSpectrogramFromSignal(signal):
    _, _, sg = scipy.signal.spectrogram(signal, fs=300, window=('tukey', 0.25), 
            nperseg=64, noverlap=0.5, return_onesided=True)

    return sg.T

def FFT(dataset):

    signals = dataset['signal']
    labels = dataset['label']

    print('Computing FFTs ...')
    logSpectrograms = []

    for s in signals:

        logSp = np.log(GenerateSpectrogramFromSignal(s) + 1)
        if logSp.shape[0] != 140:
            print('Shape diverso da 140')
            continue
        logSpectrograms.append(logSp)

    logSpectrograms = np.array(logSpectrograms)
    means = logSpectrograms.mean(axis=(1,2))
    stds = logSpectrograms.std(axis=(1,2))
    logSpectrograms = np.array([(log - mean) / std for log, mean, std in zip(logSpectrograms, means, stds)])
    logSignals = logSpectrograms[..., np.newaxis]
    
    print('Storing log signals to file..')
    with open ('./LogSignals.pk1', 'wb') as f:
        pickle.dump(logSignals, f)

    print('Done.')

    return logSignals, labels

def LoadSignalsAndLabelsFromFile(folderPath):

    signals = []
    recordsFilePath = folderPath + 'REFERENCE.csv'
    recordsAndLabels = pd.read_csv(recordsFilePath, header=None, names=['filename', 'label'])

    if os.path.isfile("./RawSignals.pk1") == False:

        print('Getting raw signals ...')
        for recordName in recordsAndLabels['filename']:     
            recordName = folderPath + recordName
            record = wfdb.rdrecord(recordName)
            digitalSignal = record.adc()[:,0]
            signals.append(digitalSignal)

        print('Done.')
        print('Saving signals to file ...')
        with open ('./RawSignals.pk1', 'wb') as f:
            pickle.dump(signals, f)
    else:
        print('Loading raw signals ...')

        with open('./RawSignals.pk1', 'rb') as fp:
            signals = pickle.load(fp)
            print('Done.') 

    signals = np.array(signals)
    y = recordsAndLabels['label']

    return signals, y

def CreateTrainTestFeatureSets(qAmplitudes, rAmplitudes, heartRatesDuringHeartBeat, rrIntervals, qrsDurations, labels, batch_size, minThresholdSignals):

    batch = []
    inputs = []

    for i in range(batch_size):

        inputs = np.vstack((rAmplitudes[i, :minThresholdSignals], qAmplitudes[i, :minThresholdSignals]))
        inputs = np.vstack((inputs, heartRatesDuringHeartBeat[i, :minThresholdSignals]))
        inputs = np.vstack((inputs, rrIntervals[i, :minThresholdSignals]))
        
        batch.append(inputs.reshape(4,minThresholdSignals))
         
        inputs = []

        y = labels[:batch_size]
        y[y=='N'] = 0
        y[y=='A'] = 1
        y[y=='O'] = 2
        y[y=='~'] = 3
        y = keras.utils.to_categorical(y)

    return batch, y

def BaselineWanderFilter(signals):

    print('Filtering ...')
    # Sampling frequency
    fs = 300

    for i, signal in enumerate(signals):

        # Baseline estimation
        win_size = int(np.round(0.2 * fs)) + 1
        baseline = medfilt(signal, win_size)
        win_size = int(np.round(0.6 * fs)) + 1
        baseline = medfilt(baseline, win_size)

        # Removing baseline
        filt_data = signal - baseline
        signals[i] = filt_data
    
    print('Storing filtered signals..')
    with open ('./FilteredSignals.pk1', 'wb') as f:
        pickle.dump((signals), f)
    print('Done.')

    return signals

def NormalizeData(signals, labels):

    print('Normalizing data ...')
    for i, signal in enumerate(signals):

        # Amplitude estimate
        norm_factor = np.percentile(signal, 99) - np.percentile(signal, 5)
        signals[i] = signal / norm_factor

    print('Done.')
    return signals, labels

def RandomCrop(df, target_size=9000, center_crop=True):
    
    signals = df['signal']
    labels = df['label']
    newSignals = []
    print('Cropping data ...')

    for i, data in enumerate(signals):

        N = data.shape[0]
        
        # Return data if correct size
        if N == target_size:
            newSignals.append(data)
            continue
        
        # If data is too small, then pad with zeros
        if N < target_size:
            tot_pads = target_size - N
            left_pads = int(np.ceil(tot_pads / 2))
            right_pads = int(np.floor(tot_pads / 2))
            newSignal = np.pad(data, [left_pads, right_pads], mode='constant')
            newSignals.append(newSignal)
            continue

        # Random Crop (always centered if center_crop=True)
        if center_crop:
            from_ = int((N / 2) - (target_size / 2))
        else:
            from_ = np.random.randint(0, np.floor(N - target_size))
        to_ = from_ + target_size
        newSignal = data[from_:to_]
        newSignals.append(newSignal)
    
    dataset = {

        'signal' : newSignals,
        'label' : labels
    }

    dataset = pd.DataFrame(dataset, columns = ['signal', 'label'])

    print('Done.')
    return dataset

def ExtractFeatures(dataset, size, minThreshold):
    # Features
    qAmplitudes = []
    rAmplitudes = []
    qrsDurations = []
    rrIntervals = []
    heartRatesDuringHeartBeat = []
    minSize = sys.maxsize

    signals = dataset['signal']
    labels = dataset['label']

    for i, sig in enumerate(signals[:size]):

        print(str(i) + "/" + str(len(signals)) + " ...")
        qrsInds = processing.gqrs_detect(sig = sig.astype('float64'), fs = 300)

        heartRates = processing.compute_hr(sig_len = sig.shape[0], fs = 300, qrs_inds = sorted(qrsInds))

        rPoints, sPoints, qPoints = QRS_util.ECG_QRS_detect(sig, 300, True, False)
            
        lenHb = sys.maxsize
        lenSigQrs = sys.maxsize
        lenHr = sys.maxsize
        lenQ = sys.maxsize
        lenR = sys.maxsize

        if(len(heartRates) > 0 and len(rPoints) > 0):
        # Adding features for each signal.
            heartRatesDuringHeartBeat.append(np.array(heartRates[rPoints]))
            lenHr = len(heartRates[rPoints])
       
        if(len(qPoints) > 0):
            qAmplitudes.append(np.array(sig[qPoints]))
            lenQ = len(sig[qPoints])
        
        if(len(rPoints) > 0):
            rAmplitudes.append(np.array(sig[rPoints]))
            lenR = len(sig[rPoints])

        if(len(qPoints) > 0 and len(rPoints) > 0):
            sigQrsDuration = [sPoints[i] - qPoints[i] for i in range(len(qPoints) - 1)]
            qrsDurations.append(np.array(sigQrsDuration))
            lenSigQrs = len(sigQrsDuration)

            hbIntervals = [rPoints[i+1] - rPoints[i] for i in range(len(rPoints) - 1)]
            rrIntervals.append(np.array(hbIntervals))
            lenHb = len(hbIntervals)

        minIter = min(lenHb, lenQ, lenR, lenSigQrs, lenHr)

        if minIter < minThreshold:
            # Rollback
            del heartRatesDuringHeartBeat[-1]
            del qAmplitudes[-1]
            del rAmplitudes[-1]
            del qrsDurations[-1]
            del rrIntervals[-1]

        elif minIter < minSize:
            minSize = minIter

    qAmplitudes = np.array([np.array(x[:minSize], dtype='int64') for x in qAmplitudes])
    rAmplitudes = np.array([np.array(x[:minSize], dtype='int64') for x in rAmplitudes])
    rrIntervals = np.array([np.array(x[:minSize], dtype='int64') for x in rrIntervals])
    qrsDurations = np.array([np.array(x[:minSize], dtype='int64') for x in qrsDurations])
    heartRatesDuringHeartBeat = np.array([np.array(x[:minSize], dtype='int64') for x in heartRatesDuringHeartBeat])

    qAmplitudes = np.stack(qAmplitudes, axis = 0)
    rAmplitudes = np.stack(rAmplitudes, axis = 0)
    heartRatesDuringHeartBeat = np.stack(heartRatesDuringHeartBeat, axis = 0)
    rrIntervals = np.stack(rrIntervals, axis = 0)
    qrsDurations = np.stack(qrsDurations, axis = 0)

    return qAmplitudes, rAmplitudes, heartRatesDuringHeartBeat, rrIntervals, qrsDurations, labels[:size], minSize

def PreprocessingForSpectrogramApproach():
    signals, labels = LoadSignalsAndLabelsFromFile(folderPath)  

    if os.path.isfile('./FilteredSignals.pk1'):
        print('Loading previously filtered signals ...')
        with open('./FilteredSignals.pk1', 'rb') as fp:
            signals = pickle.load(fp)
    else:
        signals = BaselineWanderFilter(signals)

    trainingSet, testSet = TrainTestSplit(signals, labels)
    trainingSet = TrainingTestAugumentation(trainingSet)
    trainingSet = RandomCrop(trainingSet, center_crop = False)
    trainSignals, labels = FFT(trainingSet)
    trainSignals, traininglabels = NormalizeData(trainSignals, labels) 
    
    testSet = RandomCrop(testSet, center_crop = True)
    testSignals, labels = FFT(testSet)
    testSignals, testLabels = NormalizeData(testSignals, labels) 

    print('Storing training set to file..')
    with open('./TrainingSetFFT.pk1', 'wb') as f:
        print('Len  train:' + str(len(trainSignals)))
        pickle.dump((trainSignals, traininglabels), f)
            
    print('Storing test set to file..')
    with open ('./TestSetFFT.pk1', 'wb') as f:
        print('Len  test:' + str(len(testSignals)))
        pickle.dump((testSignals, testLabels), f)

def PreprocessingForFeatureBasedApproach():

    qAmplitudes = None
    rAmplitudes = None 
    heartRatesDuringHeartBeat = None
    rrIntervals = None
    qrsDurations = None

    signals, labels = LoadSignalsAndLabelsFromFile(folderPath)  

    if os.path.isfile('./FilteredSignals.pk1'):
        print('Loading previously filtered signals ...')
        with open('./FilteredSignals.pk1', 'rb') as fp:
            signals = pickle.load(fp)
    else:
        signals = BaselineWanderFilter(signals)

    trainingSet, testSet = TrainTestSplit(signals, labels)
    trainingSet = TrainingTestAugumentation(trainingSet)
    trainingSet = RandomCrop(trainingSet, center_crop = False)
    
    testSet = RandomCrop(testSet, center_crop = True)
    
    inputSize = 0

    print('Extracting training signals features ...')
    if os.path.isfile('./TrainingFeatures.pk1'):
        with open('./TrainingFeatures.pk1', 'rb') as fp:
            qAmplitudes, rAmplitudes, heartRatesDuringHeartBeat, rrIntervals, qrsDurations, labels, inputSize = pickle.load(fp)
    else:
        qAmplitudes, rAmplitudes, heartRatesDuringHeartBeat, rrIntervals, qrsDurations, labels, minSize = ExtractFeatures(trainingSet, size = len(trainingSet), minThreshold = 20)
        with open ('./TrainingFeatures.pk1', 'wb') as f:
            pickle.dump((qAmplitudes, rAmplitudes, heartRatesDuringHeartBeat, rrIntervals, qrsDurations, labels, minSize), f)
            print('Done.')
        inputSize = minSize
    print('Done.')

    print('Storing training signal features to file..')
    with open ('./TrainingSignalFeatures.pk1', 'wb') as f:
        xTrain, yTrain = CreateTrainTestFeatureSets(qAmplitudes, rAmplitudes, heartRatesDuringHeartBeat, rrIntervals, qrsDurations, labels, len(trainingSet), minThresholdSignals = inputSize)
        pickle.dump((xTrain, yTrain), f)
        print('Done.')

    print('Extracting test signals features ...')
    if os.path.isfile('./TestFeatures.pk1'):
        with open('./TestFeatures.pk1', 'rb') as fp:
            qAmplitudes, rAmplitudes, heartRatesDuringHeartBeat, rrIntervals, qrsDurations, labels, inputSize = pickle.load(fp)
    else:
        qAmplitudes, rAmplitudes, heartRatesDuringHeartBeat, rrIntervals, qrsDurations, labels, minSize = ExtractFeatures(testSet, size = len(testSet), minThreshold = inputSize)
        with open ('./TestFeatures.pk1', 'wb') as f:
            pickle.dump((qAmplitudes, rAmplitudes, heartRatesDuringHeartBeat, rrIntervals, qrsDurations, labels, minSize), f)
            print('Done.')
    print('Done.')

    print('Storing test signal features to file..')
    with open ('./TestSignalFeatures.pk1', 'wb') as f:
        xTest, yTest = CreateTrainTestFeatureSets(qAmplitudes, rAmplitudes, heartRatesDuringHeartBeat, rrIntervals, qrsDurations, labels, len(testSet), minThresholdSignals = inputSize)
        pickle.dump((xTest, yTest), f)
        print('Done.')

    print('Preprocessing completed')

    return inputSize

def TrainingTestAugumentation(trainingSet):

    y = trainingSet['label']
    signals = trainingSet['signal']
    signals = signals.to_numpy()

    print('Class distribution before doubling up ...')
    print('Normal: ', len(np.where(y == 'N')[0]))
    print('AF: ', len(np.where(y == 'A')[0]))
    print('Other: ', len(np.where(y == 'O')[0]))
    print('Noisy: ', len(np.where(y == '~')[0]))

    # Double up AF signals

    aFibMask = np.where(y == 'A')
    aFibSignals = np.array(signals[aFibMask])

    signals = np.hstack((signals, aFibSignals))
    newAfibLabels = ['A']*len(aFibSignals)
    y = np.append(y, newAfibLabels)

    print('New AF observations number: ', len(np.where(y == 'A')[0]))

    # Duplicating noisy signals to add

    noiseMask = np.where(y == '~')
    noiseSignals = np.array(signals[noiseMask])

    for i, signal in enumerate(noiseSignals):
        noiseSignals[i] = CreateNoiseVector()

    signals = np.hstack((signals, noiseSignals))
    #   signals = np.hstack((signals, noiseSignals))
    newNoiseLabels = ['~']*len(noiseSignals)
    y = np.append(y, newNoiseLabels)

    print('New Noise observations number: ', len(np.where(y == '~')[0]))
       
    print('Done.')

    dataset = {

        'signal' : signals,
        'label' : y
    }

    dataset = pd.DataFrame(dataset, columns = ['signal', 'label'])

    print('Storing signals and labels to file..')
    with open ('./TrainingTestAugmented.pk1', 'wb') as f:
        pickle.dump((dataset), f)
        
    return dataset

