from random import seed

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

class MLPStrategyService:
    def __init__(self):
        self.last = 10000000
        seed(1)

    def run_continuo(self, data):
        df = pd.DataFrame(data, columns=["open", "high", "low", "close", "volume", "timestamp"])
        df = self.preprocess(df, True)
        df_x = df.drop(columns=["volume", "timestamp"])
        train = df_x.iloc[:-1, :-1].values
        train_target = df_x.iloc[:-1, -1].values
        # model = MLPClassifier(hidden_layer_sizes=(64, 64, 64), random_state=1, max_iter=1000)
        model = MLPRegressor(hidden_layer_sizes=(64, 64, 64), random_state=1, max_iter=1000)
        model.fit(train, train_target)
        pred = model.predict(df_x.iloc[-1, :-1].values.reshape(1, -1))[0]

        atual_close = df["close"].iloc[-1]
        aux = self.last
        self.last = pred
        return pred > aux, pred
        # return int(pred)

    def run_bool(self, data):
        df = pd.DataFrame(data, columns=["open", "high", "low", "close", "volume", "timestamp"])
        df = self.preprocess(df, False)
        df_x = df.drop(columns=["volume", "timestamp"])
        train = df_x.iloc[:-1, :-1].values
        train_target = df_x.iloc[:-1, -1].values
        model = MLPClassifier(hidden_layer_sizes=(64, 64, 64), random_state=1, max_iter=1000)
        model.fit(train, train_target)
        pred = model.predict(df_x.iloc[-1, :-1].values.reshape(1, -1))[0]

        return pred, pred

    def preprocess(self, df, continuo=False):
        df['var_1'] = (df['close'] - df['open'])
        df['var_2'] = (df['open'] - df['low'])
        df['var_3'] = (df['high'] - df['close'])
        df['var_4'] = (df['high'] - df['low'])
        # df['mean'] = df['close'].mean()
        # df['std'] = df['close'].std()
        if not continuo:
            df['target'] = df['close'].shift(-1) > df['open'].shift(-1)
        else:
            df['target'] = df['close'].shift(-1)
        return df

    def test(self, data):
        df = pd.DataFrame(data, columns=["open", "high", "low", "close", "volume", "timestamp"])
        df = self.preprocess(df)
        df_x = df.drop(columns=["volume", "timestamp"])
        size = int(df_x.shape[0]*0.8)
        train_x = df_x.iloc[:size, :-1]
        train_y = df_x.iloc[:size, -1]
        test_x = df_x.iloc[size:-1, :-1]
        test_y = df_x.iloc[size:-1, -1]
        model = MLPRegressor(hidden_layer_sizes=(64, 64, 64), random_state=1, max_iter=1000)
        model.fit(train_x, train_y)
        pred = pd.DataFrame(model.predict(test_x), index=test_x.index, columns=['pred'])

        train_x['close'].plot(legend=True, label='Treino')
        test_x['close'].plot(legend=True, label='Teste')
        pred['pred'].plot(legend=True, label='Previsões MLP')
        plt.show()
        return 1

service = MLPStrategyService()
