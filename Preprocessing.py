import numpy as np
import pandas as pd
import sys
import wfdb
from scipy.signal import medfilt
import scipy
import keras
import pickle
import os.path
from tqdm import tqdm
from random import seed
from sklearn.model_selection import train_test_split

folderPath = './training2017/'
recordsPath = folderPath + 'REFERENCE.csv'
isFourierEnabled = True

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

def FindMinLen(signals):
    minLen = None

    for signal in signals:
        if minLen is None:
            minLen = signal.size

        if signal.size < minLen:
            minLen = signal.size

    return minLen

def BaselineWanderFilter(singals):

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

    return filt_data

def NormalizeData(signals):

    print('Normalizing data ...')
    for i, signal in enumerate(signals):

        # Amplitude estimate
        norm_factor = np.percentile(signal, 99) - np.percentile(signal, 5)
        signals[i] = signal / norm_factor

    print('Done.')
    return signals

def RandomCrop(signals, target_size=9000, center_crop=False):
    
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
  
if __name__ == '__main__':

    signals, labels = LoadSignalsAndLabelsFromFile(folderPath)  
    signals = BaselineWanderFilter(signals)

    if os.path.isfile('FilteredSignals.pk1'):
        signals = pickle.load('FilteredSignals.pk1', 'rb')
    
    signals = RandomCrop(signals) 

    if isFourierEnabled == True:
        signals = FFT(signals)
    
    signals = NormalizeData(signals) 
        
    if isFourierEnabled == False:
        train, test = TrainTestSplit(signals, labels, False)

        print('Storing training set to file..')
        with open ('./TrainingSet.pk1', 'wb') as f:
            pickle.dump((train), f)
            
        print('Storing test set to file..')
        with open ('./TestSet.pk1', 'wb') as f:
            pickle.dump((test), f)

    else:

        xTrain, xTest, yTrain, yTest = TrainTestSplit(signals, labels, True)

        print('Storing training set to file..')
        with open ('./TrainingSetFFT.pk1', 'wb') as f:
            pickle.dump((xTrain, yTrain), f)
            
        print('Storing test set to file..')
        with open ('./TestSetFFT.pk1', 'wb') as f:
            pickle.dump((xTest, yTest), f)

    print('Preprocessing completed')