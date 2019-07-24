# AFib Classifier from ECG signals using Spectrogram and Feature Based Approaches

The project consists of a different multiclassifier built following two approaches. The first one, the spectrogram approach, transforms signals into a spectrogram and predicts its class like it was an image. The second one relies on features of Ecgs.

## How to run

### Feature based approaches

- Feature signal preprocessing:
`python AFib.py features preprocessing`

- Feature RNN training with default epochs:
`python AFib.py features train rnn`

- Feature RNN training with given epochs:
`python AFib.py features train rnn 30`

- Feature RNN evaluation with default weights:
`python AFib.py features evaluate rnn`

- Feature RNN evaluation with default weights:
`python AFib.py features evaluate rnn ./weights.h5`

- Feature CRNN training with default epochs:
`python AFib.py features train crnn`

- Feature CRNN training with given epochs:
`python AFib.py features train crnn 30`

- Feature CRNN evaluation with default weights:
`python AFib.py features evaluate crnn`

- Feature CRNN evaluation with given weights:
`python AFib.py features evaluate crnn ./weights.h5`

### Spectrogram based approaches

- Spectrogram signal preprocessing:
`python AFib.py spectrogram preprocessing`

- Spectrogram CNN training with default epochs:
`python AFib.py spectrogram train cnn`

- Spectrogram CNN training with given epochs:
`python AFib.py spectrogram train cnn 30`

- Spectrogram CNN evaluation with default weights:
`python AFib.py spectrogram evaluate cnn`

- Spectrogram CNN evaluation with default weights:
`python AFib.py spectrogram evaluate cnn ./weights.h5`

- Spectrogram CRNN training with default epochs:
`python AFib.py spectrogram train crnn`

- Spectrogram CRNN training with given epochs:
`python AFib.py spectrogram train crnn 30`

- Spectrogram CRNN evaluation with default weights:
`python AFib.py spectrogram evaluate crnn`

- Spectrogram CRNN evaluation with given weights:
`python AFib.py spectrogram evaluate crnn ./weights.h5`
