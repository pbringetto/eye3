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
        return ['at_bollinger_low', 'below_bollinger_low', 'rsi_oversold', 'bullish_regular', 'bullish_hidden', 'macd_rising'], ['above_bollinger_high', 'at_bollinger_high', 'rsi_overbought', 'bearish_regular', 'bearish_regular', 'macd_dropping']

    def get_signals(self, pair, tf):
        return self.signal_data.get_signals(pair, tf)

    def get_recent_signals(self, pair, tf):
        signals = self.signal_data.get_signals(pair, tf)
        signals = list(filter(lambda x: x['created_at'] == signals[0]['created_at'], signals))
        return signals

    def split_list(self, list, key, key_list_1, key_list_2):
        l1, l2 = [], []
        for item in list:
            if item[key] in key_list_1:
                l1.append(item)
            if item[key] in key_list_2:
                l2.append(item)
        return l1, l2

    def split_signals(self, signals):
        buy_signal_keys, sell_signal_keys = self.signal_keys()
        buy_signals, sell_signals = self.split_list(signals, 'key', buy_signal_keys, sell_signal_keys)
        return buy_signals, sell_signals

    def structure_change(self, signals, data):
        previous_signals = list(filter(lambda x: x['created_at'] == signals[0]['created_at'], signals))
        previous_signals_dict = {}
        for value in previous_signals:
            previous_signals_dict[value['key']] = value['value']
        current_signals = {}
        for item in signals:
            if item['value']:
                current_signals[item['key']] = item['value']
        shared_items = {k: previous_signals_dict[k] for k in previous_signals_dict if k in current_signals and previous_signals_dict[k] == current_signals[k]}
        return (len(previous_signals_dict) != len(shared_items)) or len(signals) == 0
