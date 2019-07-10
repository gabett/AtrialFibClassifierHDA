import Preprocessing
import CRNN_FeatureBased
import CRNN_SpectrogramBased
import CNN_SpectrogramBased
import sys


features = True
spectrogram = False

if __name__ == "__main__":

    if features == True:

        Preprocessing.PreprocessingForFeatureBasedApproach(isFourierEnabled=False)
        model = CRNN_FeatureBased.CRNN(input_shape =  (4, 13, 1))
        CRNN_FeatureBased.TrainCRNN(model, 50)

    elif spectrogram == True:
        Preprocessing.PreprocessingForSpectrogramApproach(isFourierEnabled=True)

        # CRNN
        # model = CRNN_SpectrogramBased.CRNN(4, 6, (140, 33, 1))
        # CRNN_SpectrogramBased.TrainCRNN(model, 1)
        # CRNN_SpectrogramBased.EvaluateCRNN(model, './crnn_model.h5')

        # CNN
        model = CNN_SpectrogramBased.CNN(4, 6, (140, 33, 1))
        #CNN_SpectrogramBased.TrainCNN(model, 1)
        CNN_SpectrogramBased.EvaluateCNN(model, './cnn_model.h5')



        