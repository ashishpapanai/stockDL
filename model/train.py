import pandas as pd
import numpy as np
from pandas_datareader import data as pdr
import keras
from keras.callbacks import TensorBoard, ReduceLROnPlateau
import tensorflow as tf
import preprocessing
import models
class Training():
    def __init__(self):
        self.learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss',
                                            patience=25,
                                            verbose=1,
                                            factor=0.25,
                                            min_lr=0.00001)
        self.tensorboard = keras.callbacks.TensorBoard(
                                                        log_dir='TensorBoard_Logs',
                                                        histogram_freq=1,
                                                        embeddings_freq=1
                                                    )       
        self.lstm_history, self.mix_history = self.train_model()
        models.Models.lstm_model.save_weights("model/lstm_weights.h5")
        models.Models.mix_lstm_model.save_weights("model/mix_lstm_weights.h5")

    def train_model(self):
        lstm_history = self.lstm_model.fit(preprocessing.X_train, preprocessing.y_train, epochs=400, batch_size=24, validation_data=(preprocessing.X_test, preprocessing.y_test),
                              verbose=1, callbacks=[self.learning_rate_reduction, self.tensorboard], shuffle=False)

        mix_history = self.mix_lstm_model.fit(preprocessing.X_train, preprocessing.y_train, epochs=400, batch_size=24, validation_data=(preprocessing.X_test, preprocessing.y_test),
                                 verbose=1, callbacks=[self.learning_rate_reduction, self.tensorboard], shuffle=False)
 
        return lstm_history, mix_history

