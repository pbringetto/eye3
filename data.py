import models.signal_model as sm
import signal as s
import cfg_load
alpha = cfg_load.load('alpha.yaml')

class Data:
    def __init__(self):
        self.signal_data = sm.SignalDataModel()

    def signal_keys(self):
        return ['at_bollinger_low', 'below_bollinger_low', 'rsi_oversold', 'bullish_regular', 'bullish_hidden', 'macd_over_signal', 'macd_over_centerline', 'macd_rising'], ['above_bollinger_high', 'at_bollinger_high', 'rsi_overbought', 'bearish_regular', 'bearish_regular', 'macd_under_signal', 'macd_under_centerline', 'macd_dropping']

    def get_signals(self, pair, tf):
        return self.signal_data.get_signals(pair, tf)

    def pair_timeframe_data(self, pair, tf, data):
        signals = self.get_signals(pair['pair'], tf['seconds'])
        signals = list(filter(lambda x: x['created_at'] == signals[0]['created_at'], signals))
        signals_dict = {}
        for value in signals:
            signals_dict[value['key']] = value['value']
        return self.split_signals(signals_dict, data, pair, tf)

    def split_signals(self, signals_dict, data, pair, tf):
        buy_signal_keys, sell_signal_keys = self.signal_keys()
        for key, value in signals_dict.items():
            if value:
                if key in buy_signal_keys:
                    data[pair['label']][tf['label']]['bullish_case'].append({key : value})
                if key in sell_signal_keys:
                    data[pair['label']][tf['label']]['bearish_case'].append({key : value})

        return data

    def signals(self):
        data = {}
        for pair in alpha["pairs"]:
            data[pair['label']] = {}
            for tf in alpha["timeframes"]:
                data[pair['label']][tf['label']] = {}
                data[pair['label']][tf['label']]['bullish_case'], data[pair['label']][tf['label']]['bearish_case'] = [], []
                data = self.pair_timeframe_data(pair, tf, data)
        return self.alpha(data)

    def alpha(self, data):
        print(data)