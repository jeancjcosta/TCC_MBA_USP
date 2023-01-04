from services.lstm_strategy_service import service as service_lstm
from services.mlp_strategy_service import service as service_mlp
from services.arima_strategy_service import service as service_arima
from services.random_strategy_service import service as service_random

class Runner():
    def run_lstm_bool(self, candles):
        return service_lstm.run_bool(candles)

    def run_lstm_continuo(self, candles):
        return service_lstm.run_continuo(candles)

    def run_mlp_bool(self, candles):
        return service_mlp.run_bool(candles)

    def run_mlp_continuo(self, candles):
        return service_mlp.run_continuo(candles)

    def run_arima(self, candles):
        return service_arima.run(candles)

    def run_random(self, candles):
        return service_random.run(candles)

    def run_lstm_test(self, candles, cripto):
        return service_lstm.test(candles, cripto)

    def run_mlp_test(self, candles, cripto):
        return service_mlp.test(candles, cripto)

    def run_arima_test(self, candles):
        return service_arima.test(candles)

runner = Runner()
