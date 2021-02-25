import keras
from keras.callbacks import TensorBoard, ReduceLROnPlateau
from numpy.core.numeric import False_
import model.preprocessing
import model.models, model.checkTraining, model.train

class Training_Variables():
    def __init__(self, ticker):
        self.preprocessing = model.preprocessing.data_preprocessing(ticker)
        self.models = model.models.Models(ticker)
        self.train = model.train.Training(ticker) 
        self.y_pred_train_lstm = self.models.lstm_model.predict(self.preprocessing.X_train)
        self.y_pred_train_mix = self.models.mix_lstm_model.predict(self.preprocessing.X_train)
        self.y_pred_lstm = self.models.lstm_model.predict(self.preprocessing.X_test)
        self.y_pred_mix = self.models.mix_lstm_model.predict(self.preprocessing.X_test)