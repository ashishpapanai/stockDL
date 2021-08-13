'''
This module helps in plotting the various details related to the model and the predictions, 
This module requires the processed data from the pre-processing module, training data from the train module,
Market information from the market module, and the calculations from the calculations module. 
'''
import os

from tensorflow.python.keras.engine.training import Model  
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
import matplotlib.pyplot as plt
from . import train, preprocessing, market, calculations, models


class Plots:
    def __init__(self, ticker):
        self.preprocessing = preprocessing.data_preprocessing(ticker)
        self.train = train.Training(ticker, "no")
        self.model = models.Models(ticker)
        self.market = market.Market(ticker)
        self.calculations = calculations.Calculations()
        #self.training_plot_loss = self.plot_training_data()
        self.predictions_plot = self.plot_predictions()
        self.in_out_plot = self.in_out()
        self.comparison_plot = self.comparison_plots()
    '''
    This function will plot the training data of the model and will use loss as the metric to depict the quality of the model. 
    This can be analsysed by TensorBoard as well, as tensorflow is a dependency of the stockDL library, 
    TensorBoard provides comparison of traning sessions by comparing the traget metrics after changing the values of parameters and hyperparameters.  
    '''
    def plot_training_data(self, metric='loss', val_metric='val_loss'):
        plt.plot(self.model.lstm_model.history[metric])
        plt.plot(self.model.lstm_model.history[val_metric])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper right')
        #plt.show()
        return plt
    
    '''
    This function plots the actual financial data of a stock and the predictions by the two deep learning strategies used in the library.
    This is a sanity check for the model and will give an idea to the user about the accuracy of the models. 
    '''
    def plot_predictions(self):
        plt.figure(figsize=(30, 10))
        plt.plot(self.preprocessing.y_train, label="Actual")
        plt.plot(self.train.y_pred_train_lstm, label="Prediction by LSTM Model")
        plt.plot(self.train.y_pred_train_mix, label="Prediction by Mix-LSTM Model")
        plt.legend(fontsize=20)
        plt.grid(axis="both")
        plt.title("Actual Open Price and Pedicted Ones on train set", fontsize=20)
        # plt.show()
        return plt

    '''
    This function plots the months we would stay in the market for trading as 1's and the months we would stay out of the market as 0's.
    The plot would overlap the predictions plot thus giving the user a visualisation if it was a good choice to stay out of the market for the particular month.

    Plotting the In-Out Months: 
        In: 1 we stay in the market as the price is greater than the actual price of the stock. 
        Out: 0 we stay out of the market as the price of the stock is less than the actual price of the stock. 
    '''

    def in_out(self):
        plt.figure(figsize=(30, 10))
        plt.plot(self.preprocessing.y_test, label="Actual")
        plt.plot(self.train.y_pred_lstm, label="LSTM Predictions")
        plt.plot(self.market.v_lstm, label="In and out LSTM")
        plt.plot(self.train.y_pred_mix, label="Mix LSTM Predictions")
        plt.plot(self.market.v_mix, label="In and out Mix LSTM")
        plt.legend(fontsize=20)
        plt.grid(axis="both")
        plt.title(
            "Actual Open Price, Predicted Ones and Vectors on In and Out Moments", fontsize=20)
        # plt.show()
        return plt

    '''
    This plotting function plots a comparison plot between the traditional financial strategies and the deep learning strategies,
    thus giving the user freedom to understand and decide what strategy to go with. 
    '''
    def comparison_plots(self):
        plt.figure(figsize=(30, 10))
        plt.plot(self.calculations.gross_portfolio(self.preprocessing.test, self.preprocessing.v_bh), label="Portfolio Buy and Hold Strategy")
        plt.plot(self.calculations.gross_portfolio(self.preprocessing.test, self.preprocessing.v_ma),
                label="Portfolio Moving Average Strategy")
        plt.plot(self.calculations.gross_portfolio(self.preprocessing.test, self.market.v_lstm), label="Portfolio LSTM Model")
        plt.plot(self.calculations.gross_portfolio(self.preprocessing.test, self.market.v_mix), label="Portfolio Mix LSTM Model")
        plt.legend(fontsize=20)
        plt.grid(axis="both")
        plt.title("Gross Portfolios of three methods", fontsize=20)
        return plt
