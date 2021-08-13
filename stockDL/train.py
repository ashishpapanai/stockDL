'''
This module handles the training of the two-deep learning strategies used in this library.
It requires preprocessing and models modules and their dependencies. 
'''
from . import preprocessing, main
from . import models
from keras.callbacks import ReduceLROnPlateau
from keras.models import load_model
from keras.optimizers import Adam
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Training:
    def __init__(self, ticker, saved):
        self.preprocessing = preprocessing.data_preprocessing(ticker)
        self.models = models.Models(ticker)
        '''Prevents false minima by reducing the learning rates on plateaus. '''
        self.learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss',
                                                         patience=25,
                                                         verbose=1,
                                                         factor=0.25,
                                                         min_lr=0.00001)
        '''Uncomment and add tensorboard to callbacks to run TensorBoard for visualisation
        self.tensorboard = keras.callbacks.TensorBoard(
                                                        log_dir='TensorBoard_Logs',
                                                        histogram_freq=1,
                                                        embeddings_freq=1
                                                    )  
        '''
        if saved.lower() == "yes":
            main.trained = True
            self.models.lstm_model.load_weights('lstm_weights.h5')
            self.models.mix_lstm_model.load_weights('mix_lstm_weights.h5')
            self.y_pred_train_lstm = self.models.lstm_model.predict(self.preprocessing.X_train)
            self.y_pred_train_mix = self.models.mix_lstm_model.predict(self.preprocessing.X_train)
            self.y_pred_lstm = self.models.lstm_model.predict(self.preprocessing.X_test)
            self.y_pred_mix = self.models.mix_lstm_model.predict(self.preprocessing.X_test)
        else:
            '''Trains the data on the defined models'''
            if main.trained:
                pass
            else:
                self.train_model()
                main.trained = True
            self.y_pred_train_lstm = self.models.lstm_model.predict(self.preprocessing.X_train)
            self.y_pred_train_mix = self.models.mix_lstm_model.predict(self.preprocessing.X_train)
            self.y_pred_lstm = self.models.lstm_model.predict(self.preprocessing.X_test)
            self.y_pred_mix = self.models.mix_lstm_model.predict(self.preprocessing.X_test)

    '''Currently both models are trained together by this function. '''

    def train_model(self):
        '''Stores the history of the LSTM model. '''
        lstm_history = self.models.lstm_model.fit(self.preprocessing.X_train, self.preprocessing.y_train, epochs=400,
                                                  batch_size=48, validation_data=(self.preprocessing.X_test,
                                                                                  self.preprocessing.y_test),
                                                  verbose=2, callbacks=[self.learning_rate_reduction], shuffle=False)
        '''Stores the history of the Conv1D + LSTM Model or the Mix Model. '''
        mix_history = self.models.mix_lstm_model.fit(self.preprocessing.X_train, self.preprocessing.y_train, epochs=400,
                                                     batch_size=48, validation_data=(self.preprocessing.X_test,
                                                                                     self.preprocessing.y_test),
                                                     verbose=2, callbacks=[self.learning_rate_reduction], shuffle=False)
        self.models.lstm_model.save_weights("lstm_weights.h5")
        self.models.mix_lstm_model.save_weights("mix_lstm_weights.h5")
        return lstm_history, mix_history
