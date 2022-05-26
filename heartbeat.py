import api.ftx as f
import strategy as s
import cfg_load
import twitter as t
import data as d
import os
import numpy as np
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
        buy_signals, sell_signals = self.data.split_signals(data)
        update = self.data.structure_change(signals, data)
        return buy_signals, sell_signals, update

    def go(self):
        strategy = s.Strategy()
        ftx = f.FtxClient(alpha["ftx_key"], alpha["ftx_secret"])
        tf = 86400
        for pair in alpha["pairs"]:

            single_market = ftx.get_single_market(pair['pair'])
            price = single_market['price']
            for tf in alpha["timeframes"]:

                df = pd.DataFrame(ftx.get_historical_prices(pair['pair'], tf['seconds']))
                df.loc[len(df.index)] = [pd.to_datetime(datetime.now().strftime("%Y-%m-%dT%H:%M:%S+00:00")), 0, price, price, price, price, 0]
                data, df = strategy.setup(df, tf['seconds'], pair['pair'])
                buy_signals, sell_signals, update = self.signals(data, pair['pair'], tf['seconds'], df)

                if update:
                    self.save_data(df, pair, tf, buy_signals, sell_signals)
                    self.post_signals(df, buy_signals, sell_signals, pair, tf)

    def save_data(self, df, pair, tf, buy_signals, sell_signals):
        ma200 = 0 if np.isnan(df['ma200'].iloc[-1]) else df['ma200'].iloc[-1]
        ema200 = 0 if np.isnan(df['ma200'].iloc[-1]) else df['ma200'].iloc[-1]
        data_id = signal_data.insert_data(df['startTime'].iloc[-1], df['open'].iloc[-1], df['high'].iloc[-1], df['low'].iloc[-1], df['close'].iloc[-1], df['volume'].iloc[-1], df['ma20'].iloc[-1], df['ma50'].iloc[-1], df['ma100'].iloc[-1], ma200, df['ema20'].iloc[-1], df['ema50'].iloc[-1], df['ema100'].iloc[-1], ema200, df['std'].iloc[-1], df['bollinger_high'].iloc[-1], df['bollinger_low'].iloc[-1], df['rsi'].iloc[-1], df['MACD_12_26_9'].iloc[-1], df['MACDh_12_26_9'].iloc[-1], df['MACDs_12_26_9'].iloc[-1], df['macd_slope'].iloc[-1], df['macd_sig_slope'].iloc[-1], df['macd_hist_slope'].iloc[-1])
        for item in buy_signals + sell_signals:
            signal_data.insert_signal(pair['pair'], tf['seconds'], item['key'], item['value'], df['startTime'].iloc[-1], data_id)

    def post_signals(self, df, buy_signals, sell_signals, pair, tf):
        if buy_signals or sell_signals:
            utc_datetime = datetime.utcnow()
            data = '#' + pair['label'] + '\r\n'
            data = data + '$' + str(df['close'].iloc[-1]) + '\r\n'
            data = data + str(utc_datetime.strftime("%Y-%m-%d %H:%M:%S")) + '\r\n'
            data = data + 'Timeframe: ' + tf['label'] + '\r\n\r\n'
            self.tweet(self.signal_content(data, buy_signals, sell_signals))

    def signal_content(self, data, buy_signals, sell_signals):
        data = data + self.signal_sections(buy_signals, 'Bullish Signals')
        if buy_signals and sell_signals:
            data = data + '\r\n'
        data = data + self.signal_sections(sell_signals, 'Bearish Signals')
        return data

    def signal_sections(self, signals, description):
        data = ''
        if signals:
            data = description + ':\r\n'
            for value in signals:
                data = data + value['value'] + '\r\n'
        return data

    def tweet(self, data):
        twitter = t.Twitter()

        print(data)

        twitter.tweet(data)

hb = Heartbeat()


