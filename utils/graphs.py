from datetime import datetime

import pandas as pd
from matplotlib import pyplot as plt


def timestamp_to_data(timestamp) -> datetime:
    return datetime.fromtimestamp(int((int(timestamp - 60 * 60 * 1000) + 1) / 1000))


class Graphs:
    def plot_close(self, cripto):
        df = pd.read_csv('data/filtered/' + cripto +'USDT.csv')[['open', 'high', 'low', 'close', 'volume', 'timestamp']]
        df['date'] = df['timestamp'].apply(timestamp_to_data)
        plt.plot(df['date'], df['close'])
        # df = df.set_index('date')
        plt.title('Preço de fechamento do ' + cripto + ' ao longo do tempo.')
        plt.xlabel('Data')
        plt.ylabel('Preço (USD)')
        plt.gcf().autofmt_xdate()
        plt.grid(True)
        plt.savefig('data/plots/preco_fechamento' + cripto + '.png')
        plt.show()


graphs = Graphs()
