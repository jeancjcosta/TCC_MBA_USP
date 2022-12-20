import numpy as np

from services.runner import runner
import pandas as pd
import time
import re


def filter_data(src):
    df = pd.read_csv('data/candles/'+ src)[['open', 'high', 'low', 'close', 'volume', 'timestamp']]
    arr = df.values
    filter = []
    for i in range(0, df.shape[0]):
        if arr[i][5] > 1598919000000 and arr[i][5] < 1661990399000:
            filter.append(arr[i])
    res = pd.DataFrame(filter, columns=['open', 'high', 'low', 'close', 'volume', 'timestamp'])
    res.to_csv(src, sep=',', index=None)


if __name__ == '__main__':
    inicio = time.time()
    # for src in ['BTCUSDT.csv', 'BNBUSDT.csv', 'ETHUSDT.csv']:
    #     filter_data(src)
    df = pd.read_csv('data/candles/BNBUSDT.csv')[['open', 'high', 'low', 'close', 'volume', 'timestamp']]
    arr = df.values
    # profit_arr = []
    size = 200
    # # runner.run_arima_test(arr[:size])
    runner.run_lstm_test(arr[2200:2200+size])
    # for j in range(1):
    #     count = 0
    #     right = 0
    #     profit = 0
    #     max_profit = 0
    #     for i in range(size, df.shape[0]):
    #     # for i in range(size, 16889 + 300):
    #         if arr[i][5] > 1598919000000 and arr[i][5] < 1661990399000:
    #             res = runner.run_lstm(arr[i-size:i])
    #             if res == 1 and arr[i+1][0] < arr[i+1][3]:
    #                 right += 1
    #             if res == 1:
    #                 count += 1
    #                 profit += (arr[i+1][3] - arr[i+1][0])/arr[i+1][0]
    #                 print(i, res, profit)
    #                 max_profit = max(profit, max_profit)
    #             else:
    #                 print(i, res, profit)
    #
    #     profit_arr.append(profit)
    #     print("iter", j)
    #     print("Acurácia", right/count)
    #     print("certos", right)
    #     print("errados", count-right)
    #     print("total", count)
    #     print("profit", profit)
    #     print("max_profit", max_profit)
    #     print("")
    #
    # print("mean_profit", np.mean(profit_arr))
    fim = time.time()
    print('Tempo de execução:', fim - inicio)
    print('fim')

