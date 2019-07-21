import Preprocessing
import CRNN_FeatureBased
import RNN_FeatureBased
import CRNN_SpectrogramBased
import CNN_SpectrogramBased
import sys


features = False
spectrogram = True

if __name__ == "__main__":


    if features == True:

        columnsNumber = Preprocessing.PreprocessingForFeatureBasedApproach()
        print('Columns number for datasets: ' + str(columnsNumber))
        
        # CRNN
        model = CRNN_FeatureBased.CRNN(input_shape =  (4, columnsNumber))
        CRNN_FeatureBased.TrainCRNN(model, 50)
        CRNN_FeatureBased.EvaluateCRNN(model, './crnn_feat_model.h5')

        # RNN
        model = RNN_FeatureBased.RNN(input_shape = (4, columnsNumber))
        RNN_FeatureBased.TrainRNN(model, 50)
        RNN_FeatureBased.EvaluateRNN(model, './rnn_feat_model.h5')

    elif spectrogram == True:
        
        Preprocessing.PreprocessingForSpectrogramApproach()

        # CRNN
        model = CRNN_SpectrogramBased.CRNN(4, 6, (140, 33, 1))
        CRNN_SpectrogramBased.TrainCRNN(model, 1)
        CRNN_SpectrogramBased.EvaluateCRNN(model, './crnn_model.h5')
        
        # CNN
        model = CNN_SpectrogramBased.CNN(4, 6, (140, 33, 1))
        CNN_SpectrogramBased.TrainCNN(model, 1)
        CNN_SpectrogramBased.EvaluateCNN(model, './cnn_model.h5')



        