import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import yfinance as yf
import keras
from keras.models import Sequential
from keras.optimizers import RMSprop, Adam
from keras.layers import Dense, Dropout, Conv1D, LSTM
from keras.callbacks import TensorBoard, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import train, preprocessing, models
class Market():
    def __init__(self, ticker):
        self.train = train.Training(ticker)
        self.w_lstm = np.diff(self.train.y_pred_lstm.reshape(self.train.y_pred_lstm.shape[0]), 1)
        self.v_lstm = np.maximum(np.sign(self.w_lstm), 0)
        self.w_mix = np.diff(self.train.y_pred_mix.reshape(self.train.y_pred_mix.shape[0]), 1)
        self.v_mix = np.maximum(np.sign(self.w_mix), 0)