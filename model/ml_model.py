# Importing all necessary libraries

"""
pandas - data preprocessing and storing in dataframes.
numpy - data preprocessing and storage in numpy arrays.
datatime - to get current date and time and to convert timestamps to date time format.
time - for operations related to date and time.
matplotlib.pyplot - to plot the graphs for better visuals and understanding. 
pandas_datareader - to read the data into dataframe from online data sources.
keras - deep learning library for simplified implementation of deep learning algorithms.
sklearn.preprocessing - to preprocess the data and make it model ready. 
yfinance - to get the financial data from Yahoo Finance
"""

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

# Restricting the GPU from consuming all memory
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

# This variable stores the stockyf symbol to retrive data from Yahoo Finance
ticker = input("Please enter the stock symbol: ")
# this pandas dataframe will store the historical market data for the stock symbol


def load_data(ticker):
    stock = yf.Ticker(ticker)
    stock.info
    # get historical market data
    df = stock.history(period="max")
    df.drop("Dividends", axis=1, inplace=True)
    df.drop("Stock Splits", axis=1, inplace=True)
    return df


df = load_data(ticker)

print(df)

# Stores the starting date for the model training i.e. the date share was launched to the market
start_date = df.index[0].to_pydatetime()
# Stores the final date for the model training i.e. the date of running the analysis
end_date = datetime.date.today()

# The analysis is run based on the opening rate of the first trading day of the market.


def date_calculations():

    start_year = start_date.year
    start_month = start_date.month
    end_year = end_date.year
    end_month = end_date.month

    first_days = []  # Stores the first days of each month

    # First year: This year along with the next year will be used to calculate the moving average
    for month in range(start_month, 13):
        first_days.append(min(df[str(start_year)+"-"+str(month)].index))

    # Other years
    for year in range(start_year+1, end_year):
        for month in range(1, 13):
            first_days.append(min(df[str(year)+"-"+str(month)].index))

    # Last year
    for month in range(1, end_month+1):
        first_days.append(min(df[str(end_year)+"-"+str(month)].index))

    return first_days


first_days = date_calculations()

# this function will create a new dataframe with the opening rate of the first day of the month.
# The moving average is calculated based on two consecutive years, The first two years are not included as it might be possible that the first year is incomplete.


def monthly_df(df):

    dfm = df.resample("M").mean()
    dfm = dfm[:-1]  # As we said, we do not consider the month of end_date

    dfm["First Day Current Month"] = first_days[:-1]
    dfm["First Day Next Month"] = first_days[1:]
    dfm["First Day Current Month Opening"] = np.array(
        df.loc[first_days[:-1], "Open"])
    dfm["First Day Next Month Opening"] = np.array(
        df.loc[first_days[1:], "Open"])
    dfm["Quotient"] = dfm["First Day Next Month Opening"].divide(
        dfm["First Day Current Month Opening"])

    dfm["mv_avg_12"] = dfm["Open"].rolling(window=12).mean().shift(1)
    dfm["mv_avg_24"] = dfm["Open"].rolling(window=24).mean().shift(1)

    # we remove the first 24 months, since they do not have the 2-year moving average
    dfm = dfm.iloc[24:, :]
    return dfm


df_monthly = monthly_df(df)
print(df_monthly.head())

# Tax Rates [India]
capital_gains_tax = 0.10
broker_comission = 0.003

# Function to calculate Gross Yield in the shares


def gross_yield(df, v):
    prod = (v*df["Quotient"]+1-v).prod()
    n_years = len(v)/12
    return (prod-1)*100, ((prod**(1/n_years))-1)*100


# Function to convert a 1D vector of zeros and ones to a 2D vector of zeros and ones with the groups of ones seperated to different columns
# E.g. [0,1,1,0,1,1,1,0,1]
"""
[[0, 1, 1, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 1, 1, 1, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 1]]),
"""
# this function will be used later [read documentation for details. ]


def separate_ones(u):

    u_ = np.r_[0, u, 0]
    i = np.flatnonzero(u_[:-1] != u_[1:])
    v, w = i[::2], i[1::2]
    if len(v) == 0:
        return np.zeros(len(u)), 0

    n, m = len(v), len(u)
    # n: tells the number of columns in the new array [number of group of 1s]
    # m: tells the number of 1s in the array
    o = np.zeros(n*m, dtype=int)

    r = np.arange(n)*m
    o[v+r] = 1

    if w[-1] == m:
        o[w[:-1]+r[:-1]] = -1
    else:
        o[w+r] -= 1

    # Store the cummulative sum of the elements
    out = o.cumsum().reshape(n, -1)
    return out, n


# The following function will calculate the net yield in the share till the investment period.
def net_yield(df, v):

    n_years = len(v)/12
    # v is the months for which we are trading in market, this variable converts months to years
    w, n = separate_ones(v)
    # w and n stores separate ones as explained in the separate_ones() function
    A = (w*np.array(df["Quotient"])+(1-w)).prod(axis=1)
    # A is the product of each group of ones of 1 for df["Quotient"]
    A1p = np.maximum(0, np.sign(A-1))
    # vector of ones where the corresponding element if  A  is > 1, other are 0
    Ap = A*A1p
    # vector of elements of A > 1, other are 0
    Am = A-Ap
    # vector of elements of A <= 1, other are 0
    An = Am+(Ap-A1p)*(1-capital_gains_tax)+A1p
    prod = An.prod()*((1-broker_comission)**(2*n))

    return (prod-1)*100, ((prod**(1/n_years))-1)*100


# Creating a window of 6 months based on which the model will make predictions for the next month
def create_window(data, window_size=1):
    data_s = data.copy()
    for i in range(window_size):
        data = pd.concat([data, data_s.shift(-(i + 1))], axis=1)

    data.dropna(axis=0, inplace=True)
    return(data)

# The data is preprocessed to be in between 0 and 1, this makes the data sutiable for Recurrent Neural Network


def data_scaling(dfm):
    # scales the values to number between 0 and 1 so that it can be RNN Ready
    scaler = MinMaxScaler(feature_range=(0, 1))
    dg = pd.DataFrame(scaler.fit_transform(dfm[["High", "Low", "Open", "Close", "Volume", "First Day Current Month Opening",
                                                "mv_avg_12", "mv_avg_24", "First Day Next Month Opening"]].values))
    X = dg[[0, 1, 2, 3, 4, 5, 6, 7]]
    X = create_window(X, window)
    X = np.reshape(X.values, (X.shape[0], window+1, 8))

    y = np.array(dg[8][window:])

    return X, y

# X: Input vector     y: Output vector
# Dimensions of input data to the model (X): (Number of Months, Window + 1, number of columns/features)
# Dimensions of output data from the model (y): (number of months)


window = 5
X, y = data_scaling(df_monthly)
print(X.shape, y.shape)

# Splitting the data to training and testing data
"""
Based on the dimensions of the input and output matrices:
Training data is split to the following dimensions: 
X_train(72, 6, 8) y_train(72, )
X_test(55, 6, 8) y_test(55, )
"""

split = 72
X_train = X[:-split-1, :, :]
X_test = X[-split-1:, :, :]
y_train = y[:-split-1]
y_test = y[-split-1:]

"""
LSTM Model: 

The first deep learning model comprises the following layers:
1. LSTM layer with 300 nodes
2. Dropout layer: which discards 50% nodes to reduce overfitting
3. LSTM layer 2 with 200 nodes to improve model
4. Dropout layer 2: which discards 50% nodes to reduce overfitting
5. Dense layer with 100 nodes to reduce the number of nodes 
6. Final Dense layer with 1 node [This layer is used as output layer] for regression. 
"""


def LSTM_Model(window, features):

    model = Sequential()
    model.add(LSTM(300, input_shape=(window, features), return_sequences=True))
    model.add(Dropout(0.5))
    # there is no need to specify input_shape here
    model.add(LSTM(200,  return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(100, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(1, kernel_initializer='uniform', activation='relu'))

    model.compile(loss='mse', optimizer=Adam(lr=0.001))

    return model


"""
MIX LSTM MODEL:

Same as the original LSTM model but with improved feature extraction in data by two 1D Convolution
"""


def Mix_LSTM_Model(window, features):

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


# the variable to store the model info
lstm_model = LSTM_Model(window+1, 8)
mix_lstm_model = Mix_LSTM_Model(window+1, 8)

""" 
Reducing learning rates when the learning curve reaches a plateau,
 so that the gradient descent doesn't get struck in in an infinite maxima/minima. 
"""
learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss',
                                            patience=25,
                                            verbose=1,
                                            factor=0.25,
                                            min_lr=0.00001)

"""
TensorBoard to monitor model performance

To run tensorboard: tensorboard --logdir=TensorBoard_Logs
"""
tensorboard = keras.callbacks.TensorBoard(
    log_dir='TensorBoard_Logs',
    histogram_freq=1,
    embeddings_freq=1
)

"""
The following variable trains the model and save the training history in it. 
"""
lstm_history = lstm_model.fit(X_train, y_train, epochs=400, batch_size=24, validation_data=(X_test, y_test),
                              verbose=1, callbacks=[learning_rate_reduction, tensorboard], shuffle=False)

mix_history = mix_lstm_model.fit(X_train, y_train, epochs=400, batch_size=24, validation_data=(X_test, y_test),
                                 verbose=1, callbacks=[learning_rate_reduction, tensorboard], shuffle=False)

"""
Save the model weights after training
"""
lstm_model.save_weights("model/lstm_weights.h5")
mix_lstm_model.save_weights("model/mix_lstm_weights.h5")

"""
Function to plot the model training and test data which would be available in TensorBoard as well. 
"""


def plot_training_data(model_selected, metric):
    plt.plot(model_selected.history['loss'])
    plt.plot(model_selected.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()

#plot_training_data(lstm_model, 'accuracy')
# Predict the values of the training data for comparisons


y_pred_train_lstm = lstm_model.predict(X_train)
y_pred_train_mix = mix_lstm_model.predict(X_train)

"""
The following lines help in plotting the prediction of LSTM and Mix LSTM model
"""

plt.figure(figsize=(30, 10))
plt.plot(y_train, label="Actual")
plt.plot(y_pred_train_lstm, label="Prediction by LSTM Model")
plt.plot(y_pred_train_mix, label="Prediction by Mix-LSTM Model")
plt.legend(fontsize=20)
plt.grid(axis="both")
plt.title("Actual Open Price and Pedicted Ones on train set", fontsize=20)
plt.show()

"""
Testing the stratergy on the testing data
"""
y_pred_lstm = lstm_model.predict(X_test)
y_pred_mix = mix_lstm_model.predict(X_test)

"""
The following variables store when we stay in market and when we move out from the market. 
"""
w_lstm = np.diff(y_pred_lstm.reshape(y_pred_lstm.shape[0]), 1)
v_lstm = np.maximum(np.sign(w_lstm), 0)

w_mix = np.diff(y_pred_mix.reshape(y_pred_mix.shape[0]), 1)
v_mix = np.maximum(np.sign(w_mix), 0)

"""
Plotting the In-Out Months: 
In: 1 we stay in market as the price is greater than the actual price of the stock. 
Out: 0 we stay out of the market as the price of stock is less than the actual price of the stock. 
"""
plt.figure(figsize=(30, 10))
plt.plot(y_test, label="Actual")
plt.plot(y_pred_lstm, label="LSTM Predictions")
plt.plot(v_lstm, label="In and out LSTM")
plt.plot(y_pred_mix, label="Mix LSTM Predictions")
plt.plot(v_mix, label="In and out Mix LSTM")
plt.legend(fontsize=20)
plt.grid(axis="both")
plt.title(
    "Actual Open Price, Predicted Ones and Vectors on In and Out Moments", fontsize=20)
plt.show()

"""
The variable vb and vh stores the months for which we will trade in the market. 
"""
test = df_monthly.iloc[-split:, :]
v_bh = np.ones(test.shape[0])
v_ma = test["First Day Current Month Opening"] > test["mv_avg_12"]

"""
The following function helps in calculating the Gross Yield:
The gross yield of an investment is its profit before taxes and expenses are deducted. 
Gross yield is expressed in percentage terms. 
"""


def gross_portfolio(df, w):
    portfolio = [(w*df["Quotient"]+(1-w))[:i].prod() for i in range(len(w))]
    return portfolio


"""
The following lines helps to plot the comparison graph of the portfolio of the 3 trading stratergies used in the project. 
"""
plt.figure(figsize=(30, 10))
plt.plot(gross_portfolio(test, v_bh), label="Portfolio Buy and Hold Strategy")
plt.plot(gross_portfolio(test, v_ma),
         label="Portfolio Moving Average Strategy")
plt.plot(gross_portfolio(test, v_lstm), label="Portfolio LSTM Model")
plt.plot(gross_portfolio(test, v_mix), label="Portfolio Mix LSTM Model")
plt.legend(fontsize=20)
plt.grid(axis="both")
plt.title("Gross Portfolios of three methods", fontsize=20)
plt.show()


"""
This fincal function stores all 
"""


def result_calculations():
    print("Test period of {:.2f} years, from {} to {} \n".format(len(
v_bh)/12, str(test.loc[test.index[0], "First Day Current Month"])[:10], str(test.loc[test.index[-1], "First Day Next Month"])[:10]))

    results = pd.DataFrame({})
    results["Method"] = ["Buy and hold", "Moving average", "LSTM", "Mix"]
    vs = [v_bh, v_ma, v_lstm, v_mix]
    results["Total Gross Yield"] = [str(round(gross_yield(test, vi)[0], 2))+" %" for vi in vs]
    results["Annual Gross Yield"] = [str(round(gross_yield(test, vi)[1], 2))+" %" for vi in vs]
    results["Total Net Yield"] = [str(round(net_yield(test, vi)[0], 2))+" %" for vi in vs]
    results["Annual Net Yield"] = [str(round(net_yield(test, vi)[1], 2))+" %" for vi in vs]
    #print(results)
    return results

print(result_calculations())

