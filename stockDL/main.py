"""
This Module handles the generation of results in the CLI, based on the ticker passed as the argument to the object of the Main class. 
The execution begins by retrieving the financial data from yfinace library which is handled in the data module of stockDL.
After the data is collected the first trading day of each month is stored in a list by the date_calculation() function in the data module.
The data is then preprocessed by the preprocessor module which creates a window to run the predictions on.
It also comprises min-max scaling which reduces sudden highs and lows in the data which would have resulted in anomalies. 
The training module trains the model created in the model module on the data retrieved by the preprocessing module. 
The plots module helps in plotting the necessary graphs for better visualisation for EDA.
The result module uses the calculation and market modules to run the necessary calculations on the predictions,
and generate net and the gross yield on the predictions obtained. 
"""
import os  
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
from . import preprocessing, train, plots, results
trained = False
class Main:
    def __init__(self, ticker, saved):
        '''Ticker is a unique symbol for every stock in the share market'''
        self.ticker = ticker
        '''this module helps in loading the data by calling an instance of data.py module and also helps in data preprocessing.'''
        self.saved = saved
        self.data_preprocessor = preprocessing.data_preprocessing(ticker)
        '''This variable stores the monthly data for analysis by the model'''
        self.df_monthly = self.data_preprocessor.df_monthly
        '''This instance of train module will help in preventing the training to occur more than once thus reducing the probability of overfitting.''' 
        self.train = train.Training(ticker, saved)
        '''Trains the data on the defined models'''
        #self.train.train_model()
        '''Creates an instance of the plots module for better visualisation of training and validation data'''
        self.plots = plots.Plots(ticker)
        '''Generates a comparison plot of the 4 methos used'''
        #self.plots.comparison_plots()
        '''An instance of the result module to calculate and process the final predictions on the data by the trained model.'''
        self.results = results.Results(ticker)
        '''Stores the final result in a pandas dataframe'''
        self.result = self.results.result
        '''Converts the pandas dataframe to JSON for better utilisation by web developers'''
        self.result_json = self.result.to_json(orient="split")
        print(self.result)
