from random import seed

import pandas as pd

from statsmodels.tsa.arima_process import arma_generate_sample
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima import auto_arima
import matplotlib.pyplot as plt


class ARIMAStrategyService:
    def run(self, data):
        seed(1)
        df = pd.DataFrame(data, columns=["open", "close", "low", "high", "volume", "timestamp"])
        # df = self.preprocess(df)
        df_x = df['close']

        result = auto_arima(df_x,  start_p=0, start_q=0,
                            max_p=3, max_q=3, m=1,
                            test='adf',
                            seasonal=False,
                            start_P=0,
                            D=0,
                            d=None, trace=True,
                            error_action='ignore',   # we don't want to know if an order does not work
                            suppress_warnings=True,  # we don't want convergence warnings
                            stepwise=True)
        # print(result.summary())
        # result.plot_diagnostics(figsize=(15, 8))
        model = ARIMA(df_x, order=result.order).fit()
        res = model.forecast()

        return res[df_x.shape[0]]

    def preprocess(self, df, past_length=2):
        df['var_1'] = (df['close'] - df['open']) / df['open']
        df['var_2'] = (df['close'] - df['close'].shift(past_length)) / df['close'].shift(past_length)
        df['var_3'] = (df['close'] - df['close'].shift(2 * past_length)) / df['close'].shift(2 * past_length)
        df['var_4'] = (df['close'] - df['close'].shift(10 * past_length)) / df['close'].shift(
            10 * past_length)

        df['lh'] = (df['high'] - df['low']) / df['low']
        df['vol_var'] = (df['volume'] - df['volume'].shift(past_length)) / df['volume'].shift(past_length)

        df['stretch'] = (df['close'] - df['low']) / df['low']
        df['mean_stretch'] = df['stretch'].rolling(5 * past_length).mean()
        df['std_stretch'] = df['stretch'].rolling(5 * past_length).std()

        df['target'] = df['close'].shift(-1) > df['open'].shift(-1)
        return df

    def test(self, data, cripto):
        df = pd.DataFrame(data, columns=["open", "close", "low", "high", "volume", "timestamp"])
        df_x = df['close']
        treino = df.iloc[:int(df_x.shape[0]*0.8)]
        teste = df.iloc[int(df_x.shape[0]*0.8):]
        result = auto_arima(treino["close"],  start_p=0, start_q=0,
                            max_p=3, max_q=3, m=1,
                            test='adf',
                            seasonal=False,
                            start_P=0,
                            D=0,
                            d=None, trace=True,
                            error_action='ignore',
                            suppress_warnings=True,
                            stepwise=True)

        model = ARIMA(treino["close"], order=result.order).fit()
        other = model.forecast()
        pred = model.predict(start=len(treino), end=len(treino)+len(teste)-1, dynamic=False, typ='levels').rename('Previsões ARIMA')
        treino['close'].plot(legend=True, label='Treino')
        teste['close'].plot(legend=True, label='Teste')
        pred.plot(legend=True, figsize=(8, 6))
        plt.savefig('data/plots/graph_pred_arima_' + cripto + '.png')
        plt.clf()

service = ARIMAStrategyService()
