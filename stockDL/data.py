'''
This module helps in data collection from Yahoo Finance API with the help of yfinace library. 
The stock data is loaded from its unique stock symbol, also called ticker. 
After the data is loaded, we drop the axis which isn't being used by us in the library.
After dropping the unnecessary axis we calculate and store the first days of each trading month in a list. 
'''
import os  
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
import datetime
import yfinance as yf
import tensorflow as tf


''' 
If the GPU is available, 
It will be restricted from consuming all memory 
else, these lines are ignored. 
'''
gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print("Using GPU")


class Data_Loader():
    def __init__(self, ticker):
        self.ticker = ticker
        self.df = self.load_data()
        self.start_date = self.df.index[0].to_pydatetime()
        self.end_date = datetime.date.today()
        ''' Stores the first day of each month from the starting date of stock to the day of executing the library '''
        self.first_days = self.date_calculations()
    
    ''' This function loads the data of a particular stock on the basis of the ticker provided by the user.  '''
    def load_data(self):
        stock = yf.Ticker(self.ticker)
        stock.info
        # get historical market data
        df = stock.history(period="max")
        df.drop("Dividends", axis=1, inplace=True)
        df.drop("Stock Splits", axis=1, inplace=True)
        return df
    
    ''' This function calculates the first trading day for each month from the first day i.e. the opening day of stock, 
    and the last day i.e. the day of executing the library. '''
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