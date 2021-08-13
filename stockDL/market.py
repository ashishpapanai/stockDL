'''
This module stores the stock market variables related to the stock ticker, 
This module must be run after training the model by calling the train_model() function in the train module.
'''
import os  
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
import numpy as np
from . import train
class Market():
    def __init__(self, ticker):
        self.train = train.Training(ticker, "no")
        self.w_lstm = np.diff(self.train.y_pred_lstm.reshape(self.train.y_pred_lstm.shape[0]), 1)
        ''' This stores the predictions of the lstm model and reshapes it. '''
        self.v_lstm = np.maximum(np.sign(self.w_lstm), 0)
        ''' Stores the sign of the lstm predictions as +ve, -ve or zero which implies profit, loss and no profit no loss in the trade. '''
        self.w_mix = np.diff(self.train.y_pred_mix.reshape(self.train.y_pred_mix.shape[0]), 1)
        ''' This stores the predictions of the mix model (conv1D + LSTM) and reshapes it. '''
        self.v_mix = np.maximum(np.sign(self.w_mix), 0)
        ''' Stores the sign of the mix model predictions as +ve, -ve or zero which implies profit, loss and no profit no loss in the trade. '''