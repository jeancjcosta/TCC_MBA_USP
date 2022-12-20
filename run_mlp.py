import numpy as np

from services.runner import runner
import pandas as pd
import time
import re


def run_bool():
    for src in ['BTCUSDT.csv', 'BNBUSDT.csv', 'ETHUSDT.csv']:
        df = pd.read_csv(src)[['open', 'high', 'low', 'close', 'volume', 'timestamp']]
        arr = df.values
        size = 240
        total_profit = 0
        # runner.run_mlp_test(arr[:size])
        process = []
        count = 0
        right = 0
        profit = 0
        max_profit = 0
        for i in range(size, df.shape[0]-1):
            result_arr = arr[i]
            res, pred = runner.run_mlp_bool(arr[i-size:i])
            result_arr = np.append(result_arr, pred)
            if res == 1 and arr[i+1][0] < arr[i+1][3]:
                right += 1
            if res == 1:
                count += 1
                profit = (arr[i+1][3] - arr[i+1][0])/arr[i+1][0]
                total_profit += profit
                print(i, res, total_profit)
                max_profit = max(total_profit, max_profit)
                result_arr = np.append(result_arr, profit)
            else:
                print(i, res, total_profit)
                result_arr = np.append(result_arr, 0)
            np.append(result_arr, result_arr)
            process.append(result_arr)
        new_df = pd.DataFrame(process, columns=['open', 'high', 'low', 'close', 'volume', 'timestamp', 'pred_boolean', 'profit_boolean'])
        new_df.to_csv("MLP_BOOL_" + src, sep=',', index=None)


def run_continuo():
    for src in ['BTCUSDT.csv', 'BNBUSDT.csv', 'ETHUSDT.csv']:
        df = pd.read_csv(src)[['open', 'high', 'low', 'close', 'volume', 'timestamp']]
        arr = df.values
        size = 240
        total_profit = 0
        # runner.run_mlp_test(arr[:size])
        process = []
        count = 0
        right = 0
        profit = 0
        max_profit = 0
        for i in range(size, df.shape[0]-1):
            result_arr = arr[i]
            res, pred = runner.run_mlp_continuo(arr[i-size:i])
            result_arr = np.append(result_arr, pred)
            if res == 1 and arr[i+1][0] < arr[i+1][3]:
                right += 1
            if res == 1:
                count += 1
                profit = (arr[i+1][3] - arr[i+1][0])/arr[i+1][0]
                total_profit += profit
                print(i, res, total_profit)
                max_profit = max(total_profit, max_profit)
                result_arr = np.append(result_arr, profit)
            else:
                print(i, res, total_profit)
                result_arr = np.append(result_arr, 0)
            np.append(result_arr, result_arr)
            process.append(result_arr)
        new_df = pd.DataFrame(process, columns=['open', 'high', 'low', 'close', 'volume', 'timestamp', 'pred_continuo', 'profit_continuo'])
        new_df.to_csv("MLP_CONT_" + src, sep=',', index=None)


if __name__ == '__main__':
    inicio = time.time()

    # run_bool()
    run_continuo()

    # print("iter", j)
    # print("Acurácia", right/count)
    # print("certos", right)
    # print("errados", count-right)
    # print("total", count)
    # print("profit", profit)
    # print("max_profit", max_profit)
    # print("")

    fim = time.time()
    print('Tempo de execução:', fim - inicio)
    print('fim')