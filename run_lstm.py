import numpy as np

from services.runner import runner
import pandas as pd
import time
import re


def run_bool(segment=1):
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


def run_continuo(segment=1):
    for src in ['BNBUSDT.csv', 'BTCUSDT.csv', 'ETHUSDT.csv']:
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
        init = size + ((segment - 1) * 2000) if segment == 1 else (segment - 1) * 2000
        for i in range(init, min(shape_x - 1, segment * 2000)):
            result_arr = arr[i]
            res, pred = runner.run_lstm_continuo(arr[i-size:i])
            result_arr = np.append(result_arr, pred)
            if res == 1 and arr[i+1][0] < arr[i+1][3]:
                right += 1
            if res == 1:
                count += 1
                profit = (arr[i+1][3] - arr[i+1][0])/arr[i+1][0]
                total_profit += profit
                print(i, res, total_profit, pred, segment)
                max_profit = max(total_profit, max_profit)
                result_arr = np.append(result_arr, profit)
            else:
                print(i, res, total_profit, pred, segment)
                result_arr = np.append(result_arr, 0)
            np.append(result_arr, result_arr)
            process.append(result_arr)
        new_df = pd.DataFrame(process, columns=['open', 'high', 'low', 'close', 'volume', 'timestamp', 'pred_continuo', 'profit_continuo'])
        new_df.to_csv(str(segment)+"LSTM_CONT_" + src, sep=',', index=False)


if __name__ == '__main__':
    inicio = time.time()

    # run_bool(1)
    # run_bool(2)
    # run_bool(3)
    # run_bool(4)
    # run_bool(5)
    # run_bool(6)
    # run_bool(7)
    # run_bool(8)
    # run_bool(9)

    # run_continuo(1)
    # run_continuo(2)
    # run_continuo(3)
    # run_continuo(4)
    # run_continuo(5)
    # run_continuo(6)
    # run_continuo(7)
    # run_continuo(8)
    run_continuo(9)



    fim = time.time()
    print('Tempo de execução:', fim - inicio)
    print('fim')