import numpy as np
import pandas as pd
import sys
import wfdb
import scipy
import keras
import pickle

folderPath = './training2017/'
recordsPath = folderPath + 'REFERENCE.csv'


def GenerateSpectogramFromSignal(signal):
    _, _, sg = scipy.signal.spectrogram(signal, fs=300, window=('tukey', 0.25), 
            nperseg=64, noverlap=0.5, return_onesided=True)

    return sg.T

def FFT(signals):
    spectograms = np.apply_along_axis(GenerateSpectogramFromSignal, 1, signals)

    logSpectograms = np.log(spectograms + 1)
    means = logSpectograms.mean(axis=(1,2))
    stds = logSpectograms.std(axis=(1,2))
    logSpectograms = np.array([(log - mean) / std for log, mean, std in zip(logSpectograms, means, stds)])
    logSignals = logSpectograms[..., np.newaxis]
    
    print('Storing log signals to file..')
    with open ('./LogSignals.pk1', 'wb') as f:
        pickle.dump(logSignals, f)

    return logSignals

def LoadSignalsAndLabelsFromFile(folderPath):

    signals = []
    recordsFilePath = folderPath + 'REFERENCE.csv'
    recordsAndLabels = pd.read_csv(recordsFilePath, header=None, names=['filename', 'label'])

    for recordName in recordsAndLabels['filename']:
        recordName = folderPath + recordName
        record = wfdb.rdrecord(recordName)
        digitalSignal = record.adc()[:,0]
        signals.append(digitalSignal)
  
    y = recordsAndLabels['label']
    y[y=='N'] = 0
    y[y=='A'] = 1
    y[y=='O'] = 2
    y[y=='~'] = 3

    y = keras.utils.to_categorical(y)

    print('Storing signals and labels to file..')
    with open ('./RecordsAndLabels.pk1', 'wb') as f:
        pickle.dump((signals, y), f)

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
    
    return filt_data

def NormalizeData(signals):

    for i, signal in enumerate(signals):

        # Amplitude estimate
        norm_factor = np.percentile(signal, 99) - np.percentile(signal, 5)
        signals[i] = signal / norm_factor

    return signals

def RandomCrop(signals, target_size=9000, center_crop=False):
    
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
        
        # Random Crop (always centered if center_crop=True)
        if center_crop:
            from_ = int((N / 2) - (target_size / 2))
        else:
            from_ = np.random.randint(0, np.floor(N - target_size))
        to_ = from_ + target_size
        signals[i] = data[from_:to_]
    
    return signals
  
if __name__ == '__main__':

    signals, labels = LoadSignalsAndLabelsFromFile(folderPath)
    filteredSignals = BaselineWanderFilter(signals)
    normalizedSignals = NormalizeData(filteredSignals)
    minLen = FindMinLen(signals)

    croppedSignals = RandomCrop(normalizedSignals)
    # transformedSignals = FFT(signals)

    print('Storing log signal and labels sto file..')
    with open ('./LogSignalsAndLabels.pk1', 'wb') as f:
        pickle.dump((croppedSignals, labels), f)

    print('Preprocessing completed')