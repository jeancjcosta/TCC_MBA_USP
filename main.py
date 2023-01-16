from services.runner import runner

import glob
import math
import os
import time

import numpy as np
import pandas as pd
from scipy.stats import t
from random import randint
from utils.graphs import graphs
from utils.metrics import metrics


def filter_data(src):
    df = pd.read_csv('data/candles/' + src)[['open', 'high', 'low', 'close', 'volume', 'timestamp']]
    arr = df.values
    filter = []
    for i in range(0, df.shape[0]):
        if arr[i][5] > 1598919000000 and arr[i][5] < 1661990399000:
            filter.append(arr[i])
    res = pd.DataFrame(filter, columns=['open', 'high', 'low', 'close', 'volume', 'timestamp'])
    res.to_csv(src, sep=',', index=None)


def run_filter_data():
    for src in ['BTCUSDT.csv', 'BNBUSDT.csv', 'ETHUSDT.csv']:
        filter_data(src)


def run_plot_close():
    for src in ['BTCUSDT.csv', 'BNBUSDT.csv', 'ETHUSDT.csv']:
        graphs.plot_close(src)


def run_random_strategy():
    for src in ['BTCUSDT', 'BNBUSDT', 'ETHUSDT']:
        print(src)
        df = pd.read_csv('data/filtered/' + src + '.csv')[['open', 'high', 'low', 'close', 'volume', 'timestamp']]
        arr = df.values
        profit_arr = []
        accuracy_arr = []
        size = 240
        for j in range(1000):
            count = 0
            right = 0
            profit = 0
            max_profit = 0
            for i in range(size, df.shape[0]-1):
                res = runner.run_random(arr[i - size:i])
                if res == 1 and arr[i + 1][0] < arr[i + 1][3]:
                    right += 1
                if res == 1:
                    count += 1
                    profit += (arr[i + 1][3] - arr[i + 1][0]) / arr[i + 1][0]
                    max_profit = max(profit, max_profit)


            profit_arr.append(profit)
            accuracy_arr.append(right / count)

        print("mean_profit", np.mean(profit_arr))
        print("mean_accuracy", np.mean(accuracy_arr))
        print("")


def continuo_metrics(model):
    for src in ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']:
        print(src + ' CONTINUO')
        df = pd.read_csv('data/processed/' + model + '_CONT_' + src + '.csv')[
            ['open', 'high', 'low', 'close', 'volume', 'timestamp', 'pred_continuo', 'profit_continuo']]
        print(metrics.classification_report_continuo(df))
        print("accuracy", metrics.accuracy_continuo(df))
        print("total profit", metrics.total_profit_continuo(df))
        print("Sharpe confidence", metrics.calc_sharpe_confidence(df["profit_continuo"].values))
        print("EQM", metrics.eqm(df))
        print("")


def boolean_metrics(model):
    for src in ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']:
        print(src + ' BOOLEAN')
        df = pd.read_csv('data/processed/' + model + '_BOOL_' + src + '.csv')[
            ['open', 'high', 'low', 'close', 'volume', 'timestamp', 'pred_boolean', 'profit_boolean']]
        df['pred_boolean'] = df['pred_boolean'].apply(lambda x: round(x, 0))
        print(metrics.classification_report_boolean(df))
        print("accuracy", metrics.accuracy_boolean(df))
        print("total profit", metrics.total_profit_boolean(df))
        print("Sharpe confidence", metrics.calc_sharpe_confidence(df["profit_boolean"].values))
        print("")


def generate_test_graphs():
    for src in ['BTCUSDT', 'BNBUSDT', 'ETHUSDT']:
        df = pd.read_csv('data/processed/MLP_BOOL_' + src + '.csv')[
            ['open', 'high', 'low', 'close', 'volume', 'timestamp']]
        runner.run_lstm_test(df.values, src)
        runner.run_mlp_test(df.values, src)


if __name__ == '__main__':
    inicio = time.time()

    # run the strategies over database and generate the graphs for test and prediction
    # generate_test_graphs()
    continuo_metrics("MLP")
    boolean_metrics("MLP")
    # run_random_strategy()

    # a = abs(359.83-1312.55)/359.83 + abs(1312.45-2706.15)/1312.45 + abs(2706.15-3000.61)/2706.15 + \
    #     abs(3000.62-2686.94)/3000.62 + abs(2686.94-1941.90)/2686.94 + abs(1941.90-1328.72)/1941.90
    # print(100*a/6)
    # bnb = abs(29.20-44.30)/29.2 + abs(44.30-353.30)/44.3 + abs(353.30-387.50)/353.30 + \
    #     abs(387.50-374.30)/387.50 + abs(374.40-320.90)/374.40 + abs(321.00-284.80)/321.00
    # print(100*bnb/6)
    fim = time.time()
    print('Tempo de execução:', fim - inicio)
    print('fim')
