import models.signal_model as sm
import signal as s
import cfg_load
import os
dir = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(dir, 'alpha.yaml')
alpha = cfg_load.load(path)

class Data:
    def __init__(self):
        self.signal_data = sm.SignalDataModel()

    def signal_keys(self):
        return ['at_bollinger_low', 'below_bollinger_low', 'rsi_oversold', 'bullish_regular', 'bullish_hidden', 'macd_over_signal', 'macd_over_centerline', 'macd_rising'], ['above_bollinger_high', 'at_bollinger_high', 'rsi_overbought', 'bearish_regular', 'bearish_regular', 'macd_under_signal', 'macd_under_centerline', 'macd_dropping']

    def get_signals(self, limit):
        return self.signal_data.get_all_signals(limit)

    def get_recent_signals(self, pair, tf):
        signals = self.signal_data.get_signals(pair, tf)
        return list(filter(lambda x: x['created_at'] == signals[0]['created_at'], signals))

    def get_last_signal(self, pair, tf, signal):
        return self.signal_data.get_signal(pair, tf, signal)

    def pair_timeframe_data(self, pair, tf, data):
        signals = self.get_recent_signals(pair['pair'], tf['seconds'])
        #signals = list(filter(lambda x: x['created_at'] == signals[0]['created_at'], signals))
        signals_dict = {}
        for value in signals:
            signals_dict[value['key']] = value['value']
        return self.split_signals(signals_dict, data, pair, tf)

    def split_signals(self, signals_dict, data, pair, tf):
        buy_signal_keys, sell_signal_keys = self.signal_keys()
        for key, value in signals_dict.items():
            if value:
                if key in buy_signal_keys:
                    data[pair['pair']][tf['seconds']]['bullish']['signals'].append({'key' : key, 'value' : value})
                if key in sell_signal_keys:
                    data[pair['pair']][tf['seconds']]['bearish']['signals'].append({'key' : key, 'value' : value})
        return data

    def signals(self):
        data = {}
        for pair in alpha["pairs"]:
            data[pair['pair']] = {}
            for tf in alpha["timeframes"]:
                data[pair['pair']][tf['seconds']] = {}
                data[pair['pair']][tf['seconds']]['bullish'], data[pair['pair']][tf['seconds']]['bearish'] = {}, {}
                data[pair['pair']][tf['seconds']]['bullish']['signals'], data[pair['pair']][tf['seconds']]['bearish']['signals'] = [], []
                data[pair['pair']][tf['seconds']]['bullish']['case'], data[pair['pair']][tf['seconds']]['bearish']['case'] = [], []
                data = self.pair_timeframe_data(pair, tf, data)
        return data