import matplotlib.pyplot as plt
from . import train, preprocessing, market, calculations


class Plots:
    def __init__(self, ticker):
        self.preprocessing = preprocessing.data_preprocessing(ticker)
        self.train = train.Training(ticker)
        # self.training_variables = training_variables.Training_Variables(ticker)
        self.market = market.Market(ticker)
        self.calculations = calculations.Calculations()

    def plot_training_data(self, model_selected, metric='loss', val_metric='val_loss'):
        plt.plot(model_selected.history[metric])
        plt.plot(model_selected.history[val_metric])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper right')
        plt.show()
    
    def plot_predictions(self):
        plt.figure(figsize=(30, 10))
        plt.plot(self.preprocessing.y_train, label="Actual")
        plt.plot(self.train.y_pred_train_lstm, label="Prediction by LSTM Model")
        plt.plot(self.train.y_pred_train_mix, label="Prediction by Mix-LSTM Model")
        plt.legend(fontsize=20)
        plt.grid(axis="both")
        plt.title("Actual Open Price and Pedicted Ones on train set", fontsize=20)
        plt.show()

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
        plt.show()

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
        plt.show()
