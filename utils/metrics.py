from typing import Tuple, Any

import numpy as np
import pandas as pd
from scipy.stats import t
from random import randint
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

class Metrics:
    # Erro quadrático médio
    def eqm(self, df) -> int:
        eqm = (df['pred_continuo'] - df['close'].shift(-1))**2
        return eqm[:-1].mean()

    # Índice de sharpe
    def calc_confidence(self, arr, confidence=0.95) -> tuple[Any, Any, Any, Any, Any]:
        m = arr.mean()
        s = arr.std()
        dof = len(arr) - 1
        t_crit = np.abs(t.ppf((1 - confidence) / 2, dof))
        return arr.min(), m - s * t_crit / np.sqrt(len(arr)), m, m + s * t_crit / np.sqrt(len(arr)), arr.max()

    def calc_sharpe(self, arr) -> int:
        return np.array(arr).mean() / np.array(arr).std()

    def calc_sharpe_confidence(self, arr, sample_size=240) -> tuple[Any, Any, Any, Any, Any]:
        sharp_array = []
        for i in range(1000):
            value = randint(sample_size, len(arr) - 1)
            sharp_array.append(self.calc_sharpe(arr[value - sample_size:value]))
        return self.calc_confidence(np.array(sharp_array), 0.95)

    # Lucro total
    def total_profit_continuo(self, df) -> int:
        return df['profit_continuo'].sum()

    def total_profit_boolean(self, df) -> int:
        return df['profit_boolean'].sum()

    # Acurácia
    def accuracy_continuo(self, df) -> int:
        return df[df['profit_continuo'] > 0].shape[0]/df[df['profit_continuo'] != 0].shape[0]

    def accuracy_boolean(self, df) -> int:
        return df[df['profit_boolean'] > 0].shape[0]/df[df['profit_boolean'] != 0].shape[0]

    # Precisão
    def classification_report_continuo(self, df):
        y_right = df['close'].shift(-1) > df['open'].shift(-1)
        y_pred = df['pred_continuo'].shift(-1) > df['pred_continuo']
        return classification_report(y_right[:-1], y_pred[:-1])

    def classification_report_boolean(self, df):
        y_right = df['close'].shift(-1) > df['open'].shift(-1)
        y_pred = df['pred_boolean']
        return classification_report(y_right[:-1], y_pred[:-1])
    # Revocação

    # F1 score

    #
metrics = Metrics()
