import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []

    for idx, item in enumerate(series[window_size:]):
        start = idx
        end = idx + window_size
        window = series[start:end]
        X.append(window)
        y.append(item)

    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()

    model.add(LSTM(5, input_shape=(window_size, 1)))

    model.add(Dense(units=1))

    return model


### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    from string import ascii_letters

    non_valid_chars = set(text) - set(ascii_letters)

    for non_valid_char in non_valid_chars:
        text = text.replace(non_valid_char, ' ')

    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs, outputs = window_transform_series(text, window_size)
    return list(inputs)[::step_size], [item[0] for item in list(outputs)[::step_size]]


# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    input_shape = (window_size, num_chars)
    model.add(LSTM(200, input_shape=input_shape))
    model.add(Dense(units=num_chars))
    model.add(Activation('softmax'))

    return model
