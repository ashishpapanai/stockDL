'''
This module makes the data ready for predictions and calculations,
This module requires the data module to get the data from the Yahoo finance API.
This module has 3 main components:
1. Convert the daily stock data to monthly data for analysis based on the opening prices for the month. 
2. Creating an analysis window for predictions. [6 years is the window size in stockDL]
3. Making data RNN ready.
'''
import os  
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from . import data


class data_preprocessing:
    def __init__(self, ticker):
        self.data_reader = data.Data_Loader(ticker)
        self.df_monthly = self.monthly_df(self.data_reader.df)
        self.window = 5
        self.X, self.y = self.data_scaling(self.df_monthly)
        self.split = 72
        self.X_train = self.X[:-self.split-1, :, :]
        self.X_test = self.X[-self.split-1:, :, :]
        self.y_train = self.y[:-self.split-1]
        self.y_test = self.y[-self.split-1:]
        self.test = self.df_monthly.iloc[-self.split:, :]
        self.v_bh = np.ones(self.test.shape[0])
        self.v_ma = self.test["First Day Current Month Opening"] > self.test["mv_avg_12"]
    '''
    This function converts the daily stock data to montly by considering the first day of the month as the day to decide
    if we should stay in the market or out. 
    It also creates moving average of the data for 12 and 24 months to device more insights for traditional investment stratergy.
    '''
    def monthly_df(self, df):

        dfm = df.resample("M").mean()
        dfm = dfm[:-1]

        dfm["First Day Current Month"] = self.data_reader.first_days[:-1]
        dfm["First Day Next Month"] = self.data_reader.first_days[1:]
        dfm["First Day Current Month Opening"] = np.array(
            df.loc[self.data_reader.first_days[:-1], "Open"])
        dfm["First Day Next Month Opening"] = np.array(
            df.loc[self.data_reader.first_days[1:], "Open"])
        dfm["Quotient"] = dfm["First Day Next Month Opening"].divide(
            dfm["First Day Current Month Opening"])

        dfm["mv_avg_12"] = dfm["Open"].rolling(window=12).mean().shift(1)
        dfm["mv_avg_24"] = dfm["Open"].rolling(window=24).mean().shift(1)

        dfm = dfm.iloc[24:, :]
        return dfm

    '''
    This function creates a sliding window for feeding the financial data to the DL model for training and analysis,
    For stockDL we use a window of 6 past years from the current date/ year. 
    '''
    def create_window(self, data, window_size):
        data_s = data.copy()
        for i in range(window_size):
            data = pd.concat([data, data_s.shift(-(i + 1))], axis=1)

        data.dropna(axis=0, inplace=True)
        return(data)

    '''
    In this function, 
    The data is preprocessed to be in between 0 and 1, this makes the data sutiable for Recurrent Neural Network
    '''
    def data_scaling(self, dfm):
        scaler = MinMaxScaler(feature_range=(0, 1))
        dg = pd.DataFrame(scaler.fit_transform(dfm[["High", "Low", "Open", "Close", "Volume", "First Day Current Month Opening",
                                                    "mv_avg_12", "mv_avg_24", "First Day Next Month Opening"]].values))
        X = dg[[0, 1, 2, 3, 4, 5, 6, 7]]
        X = self.create_window(X, self.window)
        X = np.reshape(X.values, (X.shape[0], self.window+1, 8))
        y = np.array(dg[8][self.window:])

        return X, y
