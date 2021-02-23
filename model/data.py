import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import yfinance as yf

class Data_Loader():
    def __init__(self, ticker):
        self.ticker = ticker
        self.df = self.load_data()
        self.start_date = self.df.index[0].to_pydatetime()
        self.end_date = datetime.date.today()
        self.first_days = self.date_calculations()

    def load_data(self):
        stock = yf.Ticker(self.ticker)
        stock.info
        # get historical market data
        df = stock.history(period="max")
        df.drop("Dividends", axis=1, inplace=True)
        df.drop("Stock Splits", axis=1, inplace=True)
        return df
    
    def date_calculations(self):
        start_year = self.start_date.year
        start_month = self.start_date.month
        end_year = self.end_date.year
        end_month = self.end_date.month

        first_days = []  

        for month in range(start_month, 13):
            first_days.append(min(self.df.loc[str(start_year)+"-"+str(month)].index))

        for year in range(start_year+1, end_year):
            for month in range(1, 13):
                first_days.append(min(self.df.loc[str(year)+"-"+str(month)].index))
        
        for month in range(1, end_month+1):
            first_days.append(min(self.df.loc[str(end_year)+"-"+str(month)].index))

        return first_days