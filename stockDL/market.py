import numpy as np
from . import train
class Market():
    def __init__(self, ticker):
        self.train = train.Training(ticker)
        self.w_lstm = np.diff(self.train.y_pred_lstm.reshape(self.train.y_pred_lstm.shape[0]), 1)
        self.v_lstm = np.maximum(np.sign(self.w_lstm), 0)
        self.w_mix = np.diff(self.train.y_pred_mix.reshape(self.train.y_pred_mix.shape[0]), 1)
        self.v_mix = np.maximum(np.sign(self.w_mix), 0)