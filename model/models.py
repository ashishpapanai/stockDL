import pandas as pd
import numpy as np
from pandas_datareader import data as pdr
import yfinance as yf
import keras
from keras.models import Sequential
from keras.optimizers import RMSprop, Adam
from keras.layers import Dense, Dropout, Conv1D, LSTM
from keras.callbacks import TensorBoard, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

import preprocessing

class Models():
    def __init__(self):
        self.lstm_model = self.LSTM_Model(preprocessing.window+1, 8)
        self.mix_lstm_model = self.Mix_LSTM_Model(preprocessing.window+1, 8)

    def LSTM_Model(self, window, features):
        model = Sequential()
        model.add(LSTM(300, input_shape=(window, features), return_sequences=True))
        model.add(Dropout(0.5))
        model.add(LSTM(200,  return_sequences=False))
        model.add(Dropout(0.5))
        model.add(Dense(100, kernel_initializer='uniform', activation='relu'))
        model.add(Dense(1, kernel_initializer='uniform', activation='relu'))
        model.compile(loss='mse', optimizer=Adam(lr=0.001))

        return model

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

        model.compile(loss='mse', optimizer=Adam(lr=0.001))

        return model


    