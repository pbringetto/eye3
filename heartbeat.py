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

    def signals(self, data, pair, tf, df):
        signals = self.data.get_signals(pair, tf)
        buy_signals, sell_signals = self.data.split_signals(signals)
        update = self.data.structure_change(signals, data)
        return buy_signals, sell_signals, update

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
                df.loc[len(df.index)] = [pd.to_datetime(datetime.now().strftime("%Y-%m-%dT%H:%M:%S+00:00")), 0, price, price, price, price, 0]
                data, df = strategy.setup(df, tf['seconds'], pair['pair'])
                buy_signals, sell_signals, update = self.signals(data, pair['pair'], tf['seconds'], df)

                if update:
                    for item in buy_signals + sell_signals:
                        signal_data.insert_signal(pair['pair'], tf['seconds'], item['key'], item['value'], df['startTime'].iloc[-1], single_market['price'])
                    self.tweet(buy_signals, sell_signals)

                #u.show('Market', pair['label'])
                #u.show('Timeframe', tf['label'])
                #u.show('Price', df['close'].iloc[-1])
                #u.show('Open Price', df['open'].iloc[-1])
                #u.show('High', df['high'].iloc[-1])
                #u.show('Low', df['low'].iloc[-1])

                print('bullish case------------------------------------')
                for value in buy_signals:
                    print(value['value'])

                print('bearish case------------------------------------')
                for value in sell_signals:
                    print(value['value'])

    def tweet(self, buy_signals, sell_signals):
        twitter = t.Twitter()
        twitter.tweet('hello')

hb = Heartbeat()


