import api.ftx as f
import strategy as s
import cfg_load
import twitter as t
import data as d
import os
dir = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(dir, 'alpha.yaml')
alpha = cfg_load.load(path)
import pandas as pd
import helpers.util as u
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
import models.signal_model as sm
signal_data = sm.SignalDataModel()
import time
from datetime import datetime

class Heartbeat:
    def __init__(self):
        self.data = d.Data()
        self.go()

    def signal_keys(self):
        return ['at_bollinger_low', 'below_bollinger_low', 'rsi_oversold', 'bullish_regular', 'bullish_hidden', 'macd_rising'], ['above_bollinger_high', 'at_bollinger_high', 'rsi_overbought', 'bearish_regular', 'bearish_regular', 'macd_dropping']

    def get_signals(self, pair, tf):
        return signal_data.get_signals(pair, tf)

    def split_signals(self, signals, data, pair, tf, df):
        buy_signal_keys, sell_signal_keys = self.data.signal_keys()
        buy_signals, sell_signals = [], []
        update = False
        active_signals = []
        for key, value in data.items():
            if value:
                active_signals.append({key : value})
                if key in buy_signal_keys:
                    buy_signals.append({key : value})
                if key in sell_signal_keys:
                    sell_signals.append({key : value})
        return buy_signals, sell_signals, self.structure_change(signals, data)

    def signals(self, data, pair, tf, df):
        signals = self.get_signals(pair, tf)
        return self.split_signals(signals, data, pair, tf, df)

    def structure_change(self, signals, data):
        previous_signals = list(filter(lambda x: x['created_at'] == signals[0]['created_at'], signals))
        previous_signals_dict = {}
        for value in previous_signals:
            previous_signals_dict[value['key']] = value['value']
        current_signals = {}
        for key, value in data.items():
            if value:
                current_signals[key] = value
        shared_items = {k: previous_signals_dict[k] for k in previous_signals_dict if k in current_signals and previous_signals_dict[k] == current_signals[k]}
        return (len(previous_signals_dict) != len(shared_items)) or len(signals) == 0

    def go(self):
        strategy = s.Strategy()

        ftx = f.FtxClient(alpha["ftx_key"], alpha["ftx_secret"])
        tf = 86400
        for pair in alpha["pairs"]:
            print(pair['pair'])
            single_market = ftx.get_single_market(pair['pair'])



            price = single_market['price']
            for tf in alpha["timeframes"]:


                df = pd.DataFrame(ftx.get_historical_prices(pair['pair'], tf['seconds']))

                volume = 0
                df.loc[len(df.index)] = [pd.to_datetime(datetime.now().strftime("%Y-%m-%dT%H:%M:%S+00:00")), 0, price, price, price, price, volume]

                data, df = strategy.setup(df, tf['seconds'], pair['pair'])
                buy_signals, sell_signals, update = self.signals(data, pair['pair'], tf['seconds'], df)

                if update:
                    for item in buy_signals + sell_signals:
                        for key, value in item.items():
                            print(key)
                            signal_data.insert_signal(pair['pair'], tf['seconds'], key, value, df['startTime'].iloc[-1], single_market['price'])
                    self.tweet(buy_signals, sell_signals)

                #u.show('Market', pair['label'])
                #u.show('Timeframe', tf['label'])
                #u.show('Price', df['close'].iloc[-1])
                #u.show('Open Price', df['open'].iloc[-1])
                #u.show('High', df['high'].iloc[-1])
                #u.show('Low', df['low'].iloc[-1])

                print('bullish case------------------------------------')
                for value in buy_signals:
                    print(value)

                print('bearish case------------------------------------')
                for value in sell_signals:
                    print(value)

    def tweet(self, buy_signals, sell_signals):
        twitter = t.Twitter()
        twitter.tweet('hello')

hb = Heartbeat()


