'''
This module stores the brain of the library which is the Deep Learning model.
The two Deep Learning strategies are defined in their respective methods. 
This module requires to run the preprocessing module so that the model gets the data to work on. 
'''
import os  
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Conv1D, LSTM

from . import preprocessing


class Models():
    def __init__(self, ticker):
        self.preprocessing = preprocessing.data_preprocessing(ticker)
        self.lstm_model = self.LSTM_Model(self.preprocessing.window+1, 8)
        self.mix_lstm_model = self.Mix_LSTM_Model(self.preprocessing.window+1, 8)
    '''
    The LSTM model is based on the LSTM Network which is a modified and better implementation of RNN, 
    LSTM are free from the vanishing gradients problem and so we use them for time-series predictions. 

    LSTM Model: 
    The first deep learning model comprises the following layers:
    1. LSTM layer with 300 nodes
    2. Dropout layer: which discards 50% nodes to reduce overfitting
    3. LSTM layer 2 with 200 nodes to improve model
    4. Dropout layer 2: which discards 50% nodes to reduce overfitting
    5. Dense layer with 100 nodes to reduce the number of nodes 
    6. Final Dense layer with 1 node [This layer is used as an output layer] for regression. 
    '''
    def LSTM_Model(self, window, features):
        model = Sequential()
        model.add(LSTM(300, input_shape=(window, features), return_sequences=True))
        model.add(Dropout(0.5))
        model.add(LSTM(200,  return_sequences=False))
        model.add(Dropout(0.5))
        model.add(Dense(100, kernel_initializer='uniform', activation='relu'))
        model.add(Dense(1, kernel_initializer='uniform', activation='relu'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))

        return model

    '''
    Mix LSTM Model is an improved approach with 1D convolution layer as the LSTM layers are computationally costly, 
    We use the Conv1D layer to learn the features in the data at much less computational cost. 

    
    MIX LSTM MODEL:
    Same as the original LSTM model but with improved and computationally cheap feature extraction in data by two 1D Convolution

    '''
    def Mix_LSTM_Model(self, window, features):

        model = Sequential()
        model.add(Conv1D(input_shape=(window, features), filters=32,
                        kernel_size=2, strides=1, activation='relu', padding='same'))
        model.add(Conv1D(filters=64, kernel_size=2, strides=1,
                        activation='relu', padding='same'))
        model.add(LSTM(300, return_sequences=True))
        model.add(Dropout(0.5))
        model.add(LSTM(200,  return_sequences=False))
        model.add(Dropout(0.5))
        model.add(Dense(100, kernel_initializer='uniform', activation='relu'))
        model.add(Dense(1, kernel_initializer='uniform', activation='relu'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))

        return model
