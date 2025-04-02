import keras
import tensorflow as tf
from keras._tf_keras.keras.layers import Dense, Dropout
from keras._tf_keras.keras.models import Sequential
import matplotlib.pyplot as plt
from ml_model import ml_model as ml
from utils import evaluate_model, clean_data


class dl_model(ml):

    def __init__(self, data, features,target_column, periods_in, periods_out, sequenced=True):

        [self.x_train, self.y_train], [self.x_test, self.y_test] = \
            clean_data(data, features, target_column, sequence=sequenced, periods_in=periods_in, periods_out=periods_out, train_split=True)



    def sequential_model(self, layer_1=100, layer_2=20, layer_3=1, dropout_rate=0.2, plot_loss=True, plot_pred=True, evaluate=False, log_file='seq_mod.csv'):

        base_model = Sequential()
        base_model.add(Dense(layer_1, input_dim=self.x_train.shape,  activation='relu'))
        base_model.add(Dropout(0.2))
        base_model.add(Dense(layer_2, activation='relu'))
        base_model.add(Dense(layer_3, activation='linear'))

        base_model.compile(optimizer='adam', loss='mse')

        history = base_model.fit(self.x_train, self.y_train, epochs=25)

        self.train_predict = base_model.predict(self.x_train)
        self.test_predict = base_model.predict(self.x_test)

        if plot_loss:
            plt.plot(history.history['loss'])
            plt.title('loss:'+str(round(history.history['loss'][-1], 6)))

        if plot_pred:
            self.plot_predictions(training_pred=True, test_pred=True)

        if evaluate:
            evaluate_model(self.test_predict, self.y_test, self.feats, sorted_features=False, log_file=log_file)

        self.model = base_model

        return self


def sign_penalty(y_true, y_pred):
    penalty = 100.
    loss = tf.where(tf.less(y_true * y_pred, 0), \
        penalty * tf.square(y_true - y_pred), \
        tf.square(y_true - y_pred))

    return  tf.reduce_mean(loss, axis=-1)
