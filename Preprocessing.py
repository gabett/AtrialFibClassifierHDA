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


def TrainTestSplit(signals, labels, multiDimensionalInput = False):

    if multiDimensionalInput == False:
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

        # Count the elements in the sets
        num_train_data_normal = sum(train_reference_df['label'] == 0)
        num_train_data_afib   = sum(train_reference_df['label'] == 1)
        num_train_data_other = sum(train_reference_df['label'] == 2)
        num_train_data_noise   = sum(train_reference_df['label'] == 3)

        num_val_data_normal   = sum(val_reference_df['label'] == 0)
        num_val_data_afib     = sum(val_reference_df['label'] == 1)
        num_val_data_other = sum(val_reference_df['label'] == 2)
        num_val_data_noise   = sum(val_reference_df['label'] == 3)

        print('### TRAIN SET')
        print('\tNormal ECG: {} ({:.2f}%)'.format(num_train_data_normal, 100 * num_train_data_normal / len(train_reference_df)))
        print('\tAfib ECG: {} ({:.2f}%)'.format(num_train_data_afib, 100 * num_train_data_afib / len(train_reference_df)))
        print('\tOther ECG: {} ({:.2f}%)'.format(num_train_data_other, 100 * num_train_data_other / len(train_reference_df)))
        print('\tNoisy ECG: {} ({:.2f}%)'.format(num_train_data_noise, 100 * num_train_data_noise / len(train_reference_df)))
        
        print('### VALIDATION SET')
        print('\tNormal ECG: {} ({:.2f}%)'.format(num_val_data_normal, 100 * num_val_data_normal / len(train_reference_df)))
        print('\tAfib ECG: {} ({:.2f}%)'.format(num_val_data_afib, 100 * num_val_data_afib / len(train_reference_df)))
        print('\tOther ECG: {} ({:.2f}%)'.format(num_val_data_other, 100 * num_val_data_other / len(val_reference_df)))
        print('\tNoisy ECG: {} ({:.2f}%)'.format(num_val_data_noise, 100 * num_val_data_noise / len(val_reference_df)))

        return train_reference_df, val_reference_df
    else:
        xTrain, xTest, yTrain, yTest = train_test_split(signals, labels, test_size=.2)

        num_train_data_normal = sum(np.argmax(yTrain, axis = 1) == 0)
        num_train_data_afib   = sum(np.argmax(yTrain, axis = 1) == 1)
        num_train_data_other = sum(np.argmax(yTrain, axis = 1) == 2)
        num_train_data_noise   = sum(np.argmax(yTrain, axis = 1) == 3)

        num_val_data_normal   = sum(np.argmax(yTest, axis = 1) == 0)
        num_val_data_afib     = sum(np.argmax(yTest, axis = 1) == 1)
        num_val_data_other = sum(np.argmax(yTest, axis = 1) == 2)
        num_val_data_noise   = sum(np.argmax(yTest, axis = 1) == 3)

        print('### TRAIN SET')
        print('\tNormal ECG: {} ({:.2f}%)'.format(num_train_data_normal, 100 * num_train_data_normal / len(xTrain)))
        print('\tAfib ECG: {} ({:.2f}%)'.format(num_train_data_afib, 100 * num_train_data_afib / len(xTrain)))
        print('\tOther ECG: {} ({:.2f}%)'.format(num_train_data_other, 100 * num_train_data_other / len(xTrain)))
        print('\tNoisy ECG: {} ({:.2f}%)'.format(num_train_data_noise, 100 * num_train_data_noise / len(xTrain)))
        
        print('### VALIDATION SET')
        print('\tNormal ECG: {} ({:.2f}%)'.format(num_val_data_normal, 100 * num_val_data_normal / len(xTest)))
        print('\tAfib ECG: {} ({:.2f}%)'.format(num_val_data_afib, 100 * num_val_data_afib / len(xTest)))
        print('\tOther ECG: {} ({:.2f}%)'.format(num_val_data_other, 100 * num_val_data_other / len(xTest)))
        print('\tNoisy ECG: {} ({:.2f}%)'.format(num_val_data_noise, 100 * num_val_data_noise / len(xTest)))

        return xTrain, xTest, yTrain, yTest

def CreateNoiseVector():
    noise = np.random.normal(0, 2000, size = (1, 9000))
    return noise

def GenerateSpectrogramFromSignal(signal):
    _, _, sg = scipy.signal.spectrogram(signal, fs=300, window=('tukey', 0.25), 
            nperseg=64, noverlap=0.5, return_onesided=True)

    return sg.T

def FFT(signals):

    print('Computing FFTs ...')
    spectograms = []

    for s in signals:
        spectograms.append(GenerateSpectrogramFromSignal(s))

    spectograms = np.array(spectograms)

    logSpectograms = np.log(spectograms + 1)
    means = logSpectograms.mean(axis=(1,2))
    stds = logSpectograms.std(axis=(1,2))
    logSpectograms = np.array([(log - mean) / std for log, mean, std in zip(logSpectograms, means, stds)])
    logSignals = logSpectograms[..., np.newaxis]
    
    print('Storing log signals to file..')
    with open ('./LogSignals.pk1', 'wb') as f:
        pickle.dump(logSignals, f)

    print('Done.')
    return logSignals

def LoadSignalsAndLabelsFromFile(folderPath, isFourierEnabled = False):

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

    signals = np.hstack((signals, noiseSignals))
    newNoiseLabels = ['~']*len(noiseSignals)
    y = np.append(y, newNoiseLabels)

    print('New Noise observations number: ', len(np.where(y == '~')[0]))

    y[y=='N'] = 0
    y[y=='A'] = 1
    y[y=='O'] = 2
    y[y=='~'] = 3

    if isFourierEnabled:
        y = keras.utils.to_categorical(y)

    print('Storing signals and labels to file..')
    with open ('./RecordsAndLabels.pk1', 'wb') as f:
        pickle.dump((signals, y), f)
        
    print('Done.')
    return signals, y

def CreateTrainTestFeatureSets(qAmplitudes, rAmplitudes, heartRatesDuringHeartBeat, rrIntervals, qrsDurations, labels, batch_size, trainSplit):

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

def NormalizeData(signals):

    print('Normalizing data ...')
    for i, signal in enumerate(signals):

        # Amplitude estimate
        norm_factor = np.percentile(signal, 99) - np.percentile(signal, 5)
        signals[i] = signal / norm_factor

    print('Done.')
    return signals

def RandomCrop(signals, target_size=9000, center_crop=True):
    
    print('Cropping data ...')
    for i, data in enumerate(signals):

        N = data.shape[0]
        
        # Return data if correct size
        if N == target_size:
            continue
        
        # If data is too small, then pad with zeros
        if N < target_size:
            tot_pads = target_size - N
            left_pads = int(np.ceil(tot_pads / 2))
            right_pads = int(np.floor(tot_pads / 2))
            signals[i] = np.pad(data, [left_pads, right_pads], mode='constant')
            continue

        # Random Crop (always centered if center_crop=True)
        if center_crop:
            from_ = int((N / 2) - (target_size / 2))
        else:
            from_ = np.random.randint(0, np.floor(N - target_size))
        to_ = from_ + target_size
        signals[i] = data[from_:to_]
    
    print('Done.')
    return signals

def ExtractFeatures(signals, labels, size):
    # Features
    qAmplitudes = []
    rAmplitudes = []
    qrsDurations = []
    rrIntervals = []
    heartRatesDuringHeartBeat = []
    minSize = sys.maxsize

    for i, sig in enumerate(signals[:size]):
        print(str(i) + "/" + str(len(signals)) + " ...")
        qrsInds = processing.gqrs_detect(sig = sig, fs = 300)

        heartRates = processing.compute_hr(sig_len = sig.shape[0], fs = 300, qrs_inds = sorted(qrsInds))
        
        rPoints, sPoints, qPoints = QRS_util.ECG_QRS_detect(sig, 300, True, False)
            
        lenHb = sys.maxsize
        lenQ = sys.maxsize
        lenR = sys.maxsize
        lenSigQrs = sys.maxsize
        lenHr = sys.maxsize

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

        if minIter < minSize:
            minSize = minIter

    qAmplitudes = np.array([np.array(x[:minSize]) for x in qAmplitudes])
    rAmplitudes = np.array([np.array(x[:minSize]) for x in rAmplitudes])
    rrIntervals = np.array([np.array(x[:minSize]) for x in rrIntervals])
    qrsDurations = np.array([np.array(x[:minSize]) for x in qrsDurations])
    heartRatesDuringHeartBeat = np.array([np.array(x[:minSize]) for x in heartRatesDuringHeartBeat])

    qAmplitudes = np.stack(qAmplitudes, axis = 0)
    rAmplitudes = np.stack(rAmplitudes, axis = 0)
    heartRatesDuringHeartBeat = np.stack(heartRatesDuringHeartBeat, axis = 0)
    rrIntervals = np.stack(rrIntervals, axis = 0)
    qrsDurations = np.stack(qrsDurations, axis = 0)

    return qAmplitudes, rAmplitudes, heartRatesDuringHeartBeat, rrIntervals, qrsDurations, labels[:size]

def PreprocessingForSpectrogramApproach(isFourierEnabled = False):
    signals, labels = LoadSignalsAndLabelsFromFile(folderPath, isFourierEnabled)  

    if os.path.isfile('./FilteredSignals.pk1'):
        print('Loading previously filtered signals ...')
        with open('./FilteredSignals.pk1', 'rb') as fp:
            signals = pickle.load(fp)
    else:
        signals = BaselineWanderFilter(signals)

    signals = RandomCrop(signals) 

    signals = FFT(signals)
    
    signals = NormalizeData(signals) 
    
    xTrain, xTest, yTrain, yTest = TrainTestSplit(signals, labels, True)

    print('Storing training set to file..')
    with open('./TrainingSetFFT.pk1', 'wb') as f:
        pickle.dump((xTrain, yTrain), f)
            
    print('Storing test set to file..')
    with open ('./TestSetFFT.pk1', 'wb') as f:
        pickle.dump((xTest, yTest), f)

def PreprocessingForFeatureBasedApproach(isFourierEnabled = False):

    qAmplitudes = None
    rAmplitudes = None 
    heartRatesDuringHeartBeat = None
    rrIntervals = None
    qrsDurations = None

    signals, labels = LoadSignalsAndLabelsFromFile(folderPath, isFourierEnabled)  

    if os.path.isfile('./FilteredSignals.pk1'):
        print('Loading previously filtered signals ...')
        with open('./FilteredSignals.pk1', 'rb') as fp:
            signals = pickle.load(fp)
    else:
        signals = BaselineWanderFilter(signals)

    signals = RandomCrop(signals) 
    
    if os.path.isfile('./SignalFeatures.pk1'):
        print('Loading signals features ...')
        with open('./SignalFeatures.pk1', 'rb') as fp:
            qAmplitudes, rAmplitudes, heartRatesDuringHeartBeat, rrIntervals, qrsDurations, labels = pickle.load(fp)
    else:
        print('Extracting signals features ...')
        qAmplitudes, rAmplitudes, heartRatesDuringHeartBeat, rrIntervals, qrsDurations, labels = ExtractFeatures(signals, labels, 30)
        print('Done.')

        print('Storing signal features to file..')
        with open ('./SignalFeatures.pk1', 'wb') as f:
            pickle.dump((qAmplitudes, rAmplitudes, heartRatesDuringHeartBeat, rrIntervals, qrsDurations, labels), f)
            print('Done.')

    xTrain, yTrain, xTest, yTest = CreateTrainTestFeatureSets(qAmplitudes, rAmplitudes, heartRatesDuringHeartBeat, rrIntervals, qrsDurations, labels, 10, 0.2)
    
    print('Storing training set to file..')
    with open ('./TrainingSetFeatureBased.pk1', 'wb') as f:
        pickle.dump((xTrain, yTrain), f)
            
    print('Storing test set to file..')
    with open ('./TestSetFeatureBased.pk1', 'wb') as f:
        pickle.dump((xTest, yTest), f)

    print('Preprocessing completed')