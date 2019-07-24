import Preprocessing
import CRNN_FeatureBased
import RNN_FeatureBased
import CRNN_SpectrogramBased
import CNN_SpectrogramBased
import sys


if __name__ == "__main__":

    if len(sys.argv) < 3:
        print('Mandatory parameters missing.')
    
    approach = sys.argv[1]
    method = sys.argv[2]
    model = None
    var = None

    if len(sys.argv) >= 4:
        model = sys.argv[3]
    if len(sys.argv) >= 5:
        var = sys.argv[4]

    if approach == "features":

        columnsNumber = Preprocessing.PreprocessingForFeatureBasedApproach()
        print('Columns number for datasets: ' + str(columnsNumber))
        
        if model == "crnn":
            model = CRNN_FeatureBased.CRNN(input_shape =  (4, columnsNumber))
        
            if method == "train":
                if var is None:
                    CRNN_FeatureBased.TrainCRNN(model, 100)
                else:
                    CRNN_FeatureBased.TrainCRNN(model, int(var))      
            if method == "evaluate":
                if var is None:
                    CRNN_FeatureBased.EvaluateCRNN(model, './weights-crnn.h5')
                else:
                    CRNN_FeatureBased.EvaluateCRNN(model, var)

        if model == "rnn":

            model = RNN_FeatureBased.RNN(input_shape = (4, columnsNumber))

            if method == "train":
                if var is None:
                    RNN_FeatureBased.TrainRNN(model, 100)
                else:
                    RNN_FeatureBased.TrainRNN(model, int(var))            
            if method == "evaluate":
                if var is None:
                    RNN_FeatureBased.EvaluateRNN(model, './rnn_feat_model.h5')
                else:
                    RNN_FeatureBased.EvaluateRNN(model, var)

    elif approach == "spectrogram":
        
        if method == "preprocessing":
            Preprocessing.PreprocessingForSpectrogramApproach()

        if model == "crnn":
            model = CRNN_SpectrogramBased.CRNN(4, 6, (140, 33, 1))
        
            if method == "train":  
                if var is None:   
                    CRNN_SpectrogramBased.TrainCRNN(model, 100)
                else:
                    CRNN_SpectrogramBased.TrainCRNN(model, int(var))
        
            if method == "evaluate":
                if var is None:
                    CRNN_SpectrogramBased.EvaluateCRNN(model, './model/weights-crnn-01-0.60.h5')
                else:
                    CRNN_SpectrogramBased.EvaluateCRNN(model, var)

        if model == "cnn":
            model = CNN_SpectrogramBased.CNN(4, 6, (140, 33, 1))
        
            if method == "train":
                if var is None:
                    CNN_SpectrogramBased.TrainCNN(model, 100)
                else:
                    CNN_SpectrogramBased.TrainCNN(model, int(var))
                              
            if method == "evaluate":
                if var is None or method == "all":
                    CNN_SpectrogramBased.EvaluateCNN(model, './cnn_model.h5')
                else:
                    CNN_SpectrogramBased.EvaluateCNN(model, var)



        