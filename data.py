import models.signal_model as sm
import signal as s
import cfg_load
import os
import helpers.util as u
dir = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(dir, 'alpha.yaml')
alpha = cfg_load.load(path)

class Data:
    def __init__(self):
        self.signal_data = sm.SignalDataModel()

    def signal_keys(self):
        return (
            ['falling_wedge', 'at_bollinger_low', 'below_bollinger_low', 'rsi_oversold', 'bullish_regular', 'bullish_hidden', 'macd_rising', 'below_ma_50', 'rsi_extremely_oversold', 'bullish_macd_cross'],
            ['above_bollinger_high', 'at_bollinger_high', 'rsi_overbought', 'bearish_regular', 'bearish_regular', 'macd_dropping', 'above_ma_50', 'rsi_extremely_overbought', 'bearish_macd_cross']
        )

    def get_data(self, id):
        return self.signal_data.get_data(id)

    def get_signals(self, pair, tf):
        return self.signal_data.get_signals(pair, tf)

    def get_recent_signals(self, pair, tf):
        signals = self.signal_data.get_signals(pair, tf)
        signals = list(filter(lambda x: x['created_at'] == signals[0]['created_at'], signals))
        return signals

    def split_signals(self, signals):
        buy_signal_keys, sell_signal_keys = self.signal_keys()
        buy_signals, sell_signals = u.split_list(signals, 'key', buy_signal_keys, sell_signal_keys)
        return buy_signals, sell_signals

    def structure_change(self, signals, data):
        previous_signals = u.list_to_dict(u.filter_list(signals, 'created_at', signals[0]['created_at']), 'key', 'value') if signals else []
        current_signals = u.list_to_dict(data, 'key', 'value', True)
        print('---------------------------------------------')
        print('previous_signals')
        print(previous_signals)
        print('current_signals')
        print(current_signals)
        shared_signals = u.shared_items(previous_signals, current_signals)
        print('current_signals')
        print(shared_signals)

        print((len(previous_signals) != len(shared_signals)))
        print(len(previous_signals))
        print(len(shared_signals))

        return (len(previous_signals) != len(shared_signals)) or len(signals) == 0