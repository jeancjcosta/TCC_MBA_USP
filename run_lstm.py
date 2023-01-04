import numpy as np

from services.runner import runner
import pandas as pd
import time
import re


def run_bool(src):
    df = pd.read_csv(src)[['open', 'high', 'low', 'close', 'volume', 'timestamp']]
    arr = df.values
    size = 240
    total_profit = 0
    process = []
    count = 0
    right = 0
    profit = 0
    max_profit = 0
    shape_x = df.shape[0]
    for i in range(size, shape_x-1):
        result_arr = arr[i]
        res, pred = runner.run_lstm_bool(arr[i-size:i])
        result_arr = np.append(result_arr, pred)
        if res == 1 and arr[i+1][0] < arr[i+1][3]:
            right += 1
        if res == 1:
            count += 1
            profit = (arr[i+1][3] - arr[i+1][0])/arr[i+1][0]
            total_profit += profit
            print(i, res, total_profit, src)
            max_profit = max(total_profit, max_profit)
            result_arr = np.append(result_arr, profit)
        else:
            print(i, res, total_profit, src)
            result_arr = np.append(result_arr, 0)
        np.append(result_arr, result_arr)
        process.append(result_arr)
    new_df = pd.DataFrame(process, columns=['open', 'high', 'low', 'close', 'volume', 'timestamp', 'pred_boolean', 'profit_boolean'])
    new_df.to_csv(str(src)+"LSTM_BOOL_" + src, sep=',', index=False)

def run_bool2(segment=1):
    for src in ['BTCUSDT.csv', 'BNBUSDT.csv', 'ETHUSDT.csv']:
        df = pd.read_csv(src)[['open', 'high', 'low', 'close', 'volume', 'timestamp']]
        arr = df.values
        size = 240
        total_profit = 0
        process = []
        count = 0
        right = 0
        profit = 0
        max_profit = 0
        shape_x = df.shape[0]
        init = size + ((segment - 1) * 2000) if segment == 1 else (segment - 1) * 2000
        for i in range(init, min(shape_x-1, segment * 2000)):
            result_arr = arr[i]
            res, pred = runner.run_lstm_bool(arr[i-size:i])
            result_arr = np.append(result_arr, pred)
            if res == 1 and arr[i+1][0] < arr[i+1][3]:
                right += 1
            if res == 1:
                count += 1
                profit = (arr[i+1][3] - arr[i+1][0])/arr[i+1][0]
                total_profit += profit
                print(i, res, total_profit, segment)
                max_profit = max(total_profit, max_profit)
                result_arr = np.append(result_arr, profit)
            else:
                print(i, res, total_profit, segment)
                result_arr = np.append(result_arr, 0)
            np.append(result_arr, result_arr)
            process.append(result_arr)
        new_df = pd.DataFrame(process, columns=['open', 'high', 'low', 'close', 'volume', 'timestamp', 'pred_boolean', 'profit_boolean'])
        new_df.to_csv(str(segment)+"LSTM_BOOL_" + src, sep=',', index=False)

def run_continuo(src):
    df = pd.read_csv(src)[['open', 'high', 'low', 'close', 'volume', 'timestamp']]
    arr = df.values
    size = 240
    total_profit = 0
    # runner.run_lstm_test(arr[:size])
    process = []
    count = 0
    right = 0
    profit = 0
    max_profit = 0
    shape_x = df.shape[0]
    for i in range(size, shape_x-1):
        result_arr = arr[i]
        res, pred = runner.run_lstm_continuo(arr[i-size:i])
        result_arr = np.append(result_arr, pred)
        if res == 1 and arr[i+1][0] < arr[i+1][3]:
            right += 1
        if res == 1:
            count += 1
            profit = (arr[i+1][3] - arr[i+1][0])/arr[i+1][0]
            total_profit += profit
            print(i, res, total_profit, pred, src)
            max_profit = max(total_profit, max_profit)
            result_arr = np.append(result_arr, profit)
        else:
            print(i, res, total_profit, pred, src)
            result_arr = np.append(result_arr, 0)
        np.append(result_arr, result_arr)
        process.append(result_arr)
    new_df = pd.DataFrame(process, columns=['open', 'high', 'low', 'close', 'volume', 'timestamp', 'pred_continuo', 'profit_continuo'])
    new_df.to_csv("LSTM_CONT_" + src, sep=',', index=False)


if __name__ == '__main__':
    inicio = time.time()

    # run_bool('BTCUSDT.csv')
    # run_bool('BNBUSDT.csv')
    # run_bool('ETHUSDT.csv')
    #
    run_continuo('data/filtered/BTCUSDT.csv')
    # run_continuo('BNBUSDT.csv')
    # run_continuo('ETHUSDT.csv')



    fim = time.time()
    print('Tempo de execução:', fim - inicio)
    print('fim')