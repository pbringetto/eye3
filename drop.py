import data as d
import os
import cfg_load
dir = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(dir, 'alpha.yaml')
alpha = cfg_load.load(path)

class Drop:
    def __init__(self):
        self.data = d.Data()

    def history(self):
        data = []
        for pair in alpha["pairs"]:
            for tf in alpha["timeframes"]:
                data.append({'pair': pair, 'timeframe': tf, 'signals': self.data.get_recent_signals(pair['pair'], tf['seconds'])})
        return data

    def alpha(self):
        #macd_rising_following_oversold_divergence
        data = []
        for pair in alpha["pairs"]:
            for tf in alpha["timeframes"]:
                signals = self.data.get_recent_signals(pair['pair'], tf['seconds'])
                updated =  signals[0]['created_at'] if signals else 0
                price =  signals[0]['price'] if signals else 0
                buy_signals, sell_signals = self.data.split_signals(signals)
                data.append({'pair': pair, 'timeframe': tf, 'signals': {'buy_signals': buy_signals, 'sell_signals': sell_signals}, 'updated': updated, 'price': price})
                print(data)
        return data