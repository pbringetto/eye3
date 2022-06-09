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
from numpy import nan
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import matplotlib.lines as lines
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

class Heartbeat:
    def __init__(self):
        self.data = d.Data()
        self.go()

    def signals(self, data, pair, tf, df):
        signals = self.data.get_signals(pair, tf)
        buy_signals, sell_signals = self.data.split_signals(data)
        update = self.data.structure_change(signals, buy_signals + sell_signals)
        return buy_signals, sell_signals, update

    def go(self):
        strategy = s.Strategy()
        ftx = f.FtxClient(alpha["ftx_key"], alpha["ftx_secret"])
        #tf = 86400
        for pair in alpha["pairs"]:

            single_market = ftx.get_single_market(pair['pair'])
            price = single_market['price']
            for tf in alpha["timeframes"]:
                print(tf['label'])
                df = pd.DataFrame(ftx.get_historical_prices(pair['pair'], tf['seconds']))

                df.loc[len(df.index)] = [pd.to_datetime(datetime.now().strftime("%Y-%m-%dT%H:%M:%S+00:00")), 0, price, price, price, price, 0]

                df['x'] = pd.to_datetime(df['startTime'], errors='coerce', utc=True)
                df['x'] = df['x'].dt.strftime('%Y-%m-%d')
                print(df['x'])
                data, pattern_data, df, lows, highs = strategy.setup(df, tf, pair['pair'], 2, [2, 2], [34, 34])

                buy_signals, sell_signals, update = self.signals(data, pair['pair'], tf['seconds'], df)

                print('-------------------------------------------------')
                if buy_signals:
                    print('Buy Signals')
                    print(buy_signals)
                print(' ')
                if sell_signals:
                    print('Sell Signals')
                    print(sell_signals)
                print('-------------------------------------------------')


                print(lows[['close','volume','bollinger_high','bollinger_low','rsi','rsi_slope','macd_slope','macd_sig_slope','macd_hist_slope']][-5::])
                print(highs[['close','volume','bollinger_high','bollinger_low','rsi','rsi_slope','macd_slope','macd_sig_slope','macd_hist_slope']][-5::])


                df.loc[df.close < (df.ema20 - (df['std'] * 3)), 'signal'] = 'long'
                df.loc[(df.close > (df.ema20 + (df['std'] * 3))) & df.macd_slope.gt(250), 'signal'] = 'short'



                ax = df.set_index('x').plot(kind='line', use_index=True, y='close', color="blue")

                ax.set_title(pair['label'], color='black')
                ax.set_facecolor("gray")
                x_axis = ax.axes.get_xaxis()
                x_axis.label.set_visible(False)
                ax.tick_params(axis='x', labelrotation = -90)
                ax.tick_params(labelcolor='black')

                line = lines.Line2D([ highs.iloc[-4].name, highs.iloc[-1].name ], [  highs['close'].iloc[-4], highs['close'].iloc[-1]  ],
                                    lw=2, color='tab:orange', axes=ax)
                ax.add_line(line)
                line = lines.Line2D([ lows.iloc[-4].name, lows.iloc[-1].name ], [  lows['close'].iloc[-4], lows['close'].iloc[-1]  ],
                                    lw=2, color='tab:orange', axes=ax)
                ax.add_line(line)
            
                x = int(len(df) / 2)
                patterns = []
                for  pattern in pattern_data:
                    if  pattern['value']:
                        patterns.append(pattern['value'])

                if patterns:
                    ax.annotate(patterns[0], xy=(df.iloc[-x].name, df['close'].min()), xytext=(df.iloc[-x].name, df['close'].min()))

                df = df.dropna(subset=['signal'])
                for index, row in df.iterrows():
                    #ax.annotate(row['signal'] + '   ' + str(row['rsi_slope']), xy=(index, row['close']), xytext=(index, row['close']),
                    color = 'red' if row['signal'] == 'short' else 'green'
                    ax.annotate(row['signal'], xy=(index, row['close']), xytext=(index, row['close']), color='black',
                        arrowprops=dict(facecolor='black', shrink=0.05),
                    )

                if update:
                    file = 'img/' + str(datetime.now()) + '.png'
                    plt.savefig(file)
                    self.save_data(df, pair, tf, buy_signals, sell_signals, file)
                    self.post_signals(df, buy_signals, sell_signals, pair, tf, file)
                else:

                    print('no update')

    def save_data(self, df, pair, tf, buy_signals, sell_signals, chart_image):
        ma200 = 0 if np.isnan(df['ma200'].iloc[-1]) else df['ma200'].iloc[-1]
        ema200 = 0 if np.isnan(df['ma200'].iloc[-1]) else df['ma200'].iloc[-1]
        data_id = signal_data.insert_data(df['startTime'].iloc[-1], df['open'].iloc[-1], df['high'].iloc[-1], df['low'].iloc[-1], df['close'].iloc[-1], df['volume'].iloc[-1], df['ma20'].iloc[-1], df['ma50'].iloc[-1], df['ma100'].iloc[-1], ma200, df['ema20'].iloc[-1], df['ema50'].iloc[-1], df['ema100'].iloc[-1], ema200, df['std'].iloc[-1], df['bollinger_high'].iloc[-1], df['bollinger_low'].iloc[-1], df['rsi'].iloc[-1], df['MACD_12_26_9'].iloc[-1], df['MACDh_12_26_9'].iloc[-1], df['MACDs_12_26_9'].iloc[-1], df['macd_slope'].iloc[-1], df['macd_sig_slope'].iloc[-1], df['macd_hist_slope'].iloc[-1], chart_image)
        for item in buy_signals + sell_signals:
            signal_data.insert_signal(pair['pair'], tf['seconds'], item['key'], item['value'], df['startTime'].iloc[-1], data_id)

    def post_signals(self, df, buy_signals, sell_signals, pair, tf, file):
        if buy_signals or sell_signals:
            utc_datetime = datetime.utcnow()
            data = '#' + pair['label'] + '\r\n'
            data = data + '$' + str(df['close'].iloc[-1]) + '\r\n'
            data = data + str(utc_datetime.strftime("%Y-%m-%d %H:%M:%S")) + '\r\n'
            data = data + 'Timeframe: ' + tf['label'] + '\r\n\r\n'
            self.tweet(self.signal_content(data, buy_signals, sell_signals), file)

    def signal_content(self, data, buy_signals, sell_signals):
        data = data + self.signal_sections(buy_signals, 'Buy Signals')
        if buy_signals and sell_signals:
            data = data + '\r\n'
        data = data + self.signal_sections(sell_signals, 'Sell Signals')
        return data

    def signal_sections(self, signals, description):
        data = ''
        if signals:
            data = description + ':\r\n'
            for value in signals:
                data = data + '  ' + value['value'] + '\r\n'
        return data

    def tweet(self, data, file):
        twitter = t.Twitter()

        print(data)
        if alpha["twitter_enabled"]:
            twitter.tweet(data, file)

hb = Heartbeat()


