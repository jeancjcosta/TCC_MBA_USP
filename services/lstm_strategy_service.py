from random import seed
from matplotlib import pyplot as plt
from tensorflow import keras
import pandas as pd
import numpy as np
import tensorflow as tf


# decaimento de learning rate
def scheduler(epoch, lr):
    if (epoch+1 < 5):
        return lr
    else:
        return np.round(lr * tf.math.exp(-0.04), 9)


callbacklr = tf.keras.callbacks.LearningRateScheduler(scheduler)


class LSTMStrategyService:
    def __init__(self):
        self.time_steps = 2
        self.last = 10000000
        seed(1)

    def run_bool(self, data):
        batch_size = 40
        epochs = 70
        learning_rate = 0.01
        df = pd.DataFrame(data, columns=["open", "high", "low", "close", "volume", "timestamp"])
        train_x, train_y = self.preprocess(df, continuo=False)
        model = self.model_LSTM(features=train_x.shape[2], time_steps=self.time_steps, continuo=False)
        model.compile(loss='mae', metrics=["mean_squared_error"], optimizer=keras.optimizers.Adam(learning_rate=learning_rate))
        model.fit(train_x[:-1], train_y[:-1], epochs=epochs, batch_size=batch_size, verbose=0, shuffle=False, callbacks=[callbacklr])
        pred = model.predict(train_x[-1:])[0][0]

        return round(pred, 0), round(pred, 0)

    def run_continuo(self, data):
        batch_size = 40
        epochs = 70
        learning_rate = 0.01
        df = pd.DataFrame(data, columns=["open", "high", "low", "close", "volume", "timestamp"])
        train_x, train_y = self.preprocess(df, continuo=True)
        model = self.model_LSTM(features=train_x.shape[2], time_steps=self.time_steps, continuo=True)
        model.compile(loss='mae', metrics=["mean_squared_error"], optimizer=keras.optimizers.Adam(learning_rate=learning_rate))
        model.fit(train_x[:-1], train_y[:-1], epochs=epochs, batch_size=batch_size, verbose=0, shuffle=False, callbacks=[callbacklr])
        pred = model.predict(train_x[-1:])[0][0]
        aux = self.last
        self.last = pred
        return pred > aux, pred

    def preprocess(self, df, continuo=False):
        df['var_1'] = (df['close'] - df['open'])
        df['var_2'] = (df['open'] - df['low'])
        df['var_3'] = (df['high'] - df['close'])
        df['var_4'] = (df['high'] - df['low'])

        if continuo:
            df['target'] = df['close'].shift(-1)
        else:
            df['target'] = df['close'].shift(-1) > df['open'].shift(-1)
        df = df.drop(columns=["volume", "timestamp"])

        return self.recurrent_array(df, self.time_steps)

    def model_LSTM(self, features, time_steps, continuo=False):
        modelLSTM = keras.models.Sequential()
        modelLSTM.add(keras.layers.InputLayer((time_steps, features)))
        modelLSTM.add(keras.layers.LSTM(256, activation='relu', return_sequences=True))
        modelLSTM.add(keras.layers.LSTM(128, activation='relu', return_sequences=True))
        modelLSTM.add(keras.layers.LSTM(64, activation='relu'))
        if continuo:
            modelLSTM.add(keras.layers.Dense(1))
        else:
            modelLSTM.add(keras.layers.Dense(1, activation='sigmoid'))
        return modelLSTM

    # formato deve ser [samples, time steps, features]
    def recurrent_array(self, df, time_steps):
        train_x = df.iloc[:, :-1].values
        array = train_x

        train_y_ts = df.iloc[time_steps-1:, -1].values
        train_x_ts = []
        for i in range(0, (len(array) - time_steps)+1):
            train_x_ts.append(array[i:i + time_steps])

        train_x_ts = np.array(train_x_ts)
        # rec_array = np.reshape(array, (array.shape[0], time_steps, array.shape[1]))
        return train_x_ts, train_y_ts


    def test(self, data, cripto):
        batch_size = 40
        epochs = 70
        learning_rate = 0.01
        df = pd.DataFrame(data, columns=["open", "high", "low", "close", "volume", "timestamp"])
        data_x, data_y = self.preprocess(df, continuo=True)

        size = int(len(data_x) * 0.8)
        train_x = data_x[:size]
        test_x = data_x[size:-1]
        train_y = data_y[:size]
        test_y = data_y[size:-1]

        model = self.model_LSTM(features=train_x.shape[2], time_steps=self.time_steps, continuo=True)
        model.compile(loss='mae', metrics=["mean_squared_error"], optimizer=keras.optimizers.Adam(learning_rate=learning_rate))
        model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=1, shuffle=False, callbacks=[callbacklr])
        pred = pd.DataFrame(model.predict(test_x), index=df.iloc[size+self.time_steps:].index)

        treino = df.iloc[:size]
        teste = df.iloc[size+self.time_steps:]
        treino['close'].plot(legend=True, label='Treino')
        teste['close'].plot(legend=True, label='Teste')
        pred[0].plot(legend=True, label='Previsões LSTM', figsize=(8, 6))
        plt.xlabel('Horas')
        plt.ylabel('Preço (USD)')
        plt.savefig('data/plots/graph_pred_lstm_' + cripto + '.png')
        # plt.show()
        plt.clf()

        return 1


service = LSTMStrategyService()
