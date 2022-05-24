import pandas as pd
import pandas_ta as ta
import numpy as np
import cfg_load
from scipy.stats import linregress
from scipy.signal import argrelextrema
import math
pd.options.mode.chained_assignment = None

class Indicator:
    def __init__(self):
        self.data = []

    def on_balance_volume(self, df, n):
        """Calculate On-Balance Volume for given data.

        :param df: pandas.DataFrame
        :param n:
        :return: pandas.DataFrame
        """
        i = 0
        OBV = [0]
        while i < df.index[-1]:
            if df.loc[i + 1, 'close'] - df.loc[i, 'close'] > 0:
                OBV.append(df.loc[i + 1, 'volume'])
            if df.loc[i + 1, 'close'] - df.loc[i, 'close'] == 0:
                OBV.append(0)
            if df.loc[i + 1, 'close'] - df.loc[i, 'close'] < 0:
                OBV.append(-df.loc[i + 1, 'volume'])
            i = i + 1
        OBV = pd.Series(OBV)
        OBV_ma = pd.Series(OBV.rolling(n, min_periods=n).mean(), name='obv')
        df = df.join(OBV_ma)
        data = {
                    'on_balance_volume': False,
                }
        return data, df

    def bollinger(self, df):

        tp = (df['close'] + df['low'] + df['high'])/3
        df['std'] = tp.rolling(20).std(ddof=0)
        matp = tp.rolling(20).mean()
        df['bollinger_high'] = matp + 2*df['std']
        df['bollinger_low'] = matp - 2*df['std']

        data = {'bollinger_low': df['bollinger_low'].iloc[-1], 'bollinger_high': df['bollinger_high'].iloc[-1]}
        data = [
            {'key': 'below_bollinger_low', 'value': 'Below bollinger Low' if df['close'].iloc[-1] < df['bollinger_low'].iloc[-1] else False, 'data': data},
            {'key': 'above_bollinger_high', 'value': 'Above bollinger high' if df['close'].iloc[-1] > df['bollinger_high'].iloc[-1] else False, 'data': data},
            {'key': 'at_bollinger_low', 'value': 'At Bollinger low' if math.isclose(df['close'].iloc[-1], df['bollinger_low'].iloc[-1], abs_tol=100) else False, 'data': data},
            {'key': 'at_bollinger_high', 'value': 'At Bollinger high' if math.isclose(df['close'].iloc[-1], df['bollinger_high'].iloc[-1], abs_tol=100) else False, 'data': data},
        ]

        '''
        data = {
            'below_bollinger_low': 'Below bollinger Low' if df['close'].iloc[-1] < df['bollinger_low'].iloc[-1] else False,
            'above_bollinger_high': 'Above bollinger high' if df['close'].iloc[-1] > df['bollinger_high'].iloc[-1] else False,
            'at_bollinger_low' : 'At Bollinger low' if math.isclose(df['close'].iloc[-1], df['bollinger_low'].iloc[-1], abs_tol=100) else False,
            'at_bollinger_high' : 'At Bollinger high' if math.isclose(df['close'].iloc[-1], df['bollinger_high'].iloc[-1], abs_tol=100) else False,
        }
        '''

        return data, df


    def ma(self, df):
        df['ma20'] = df['close'].rolling(20).mean()
        df['ma50'] = df['close'].rolling(50).mean()
        df['ma100'] = df['close'].rolling(100).mean()
        df['ma200'] = df['close'].rolling(200).mean()


        data = {
            'above_ma_20': 'Above 20 period moving average' if df['close'].iloc[-1] > df['ma20'].iloc[-1] else False,
            'above_ma_50': 'Above 50 period moving average' if df['close'].iloc[-1] > df['ma50'].iloc[-1] else False,
            'above_ma_100': 'Above 100 period moving average' if df['close'].iloc[-1] > df['ma100'].iloc[-1] else False,
            'above_ma_200': 'Above 200 period moving average' if df['close'].iloc[-1] > df['ma200'].iloc[-1] else False,
            'below_ma_20': 'Below 20 period moving average' if df['close'].iloc[-1] > df['ma20'].iloc[-1] else False,
            'below_ma_50': 'Below 50 period moving average' if df['close'].iloc[-1] > df['ma50'].iloc[-1] else False,
            'below_ma_100': 'Below 100 period moving average' if df['close'].iloc[-1] > df['ma100'].iloc[-1] else False,
            'below_ma_200': 'Below 200 period moving average' if df['close'].iloc[-1] > df['ma200'].iloc[-1] else False,
        }
        return data, df


    def ema(self, df):

        df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema100'] = df['close'].ewm(span=100, adjust=False).mean()
        df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()

        data = {
            'above_ema_20': 'Above 20 period moving average' if df['close'].iloc[-1] > df['ema20'].iloc[-1] else False,
            'above_ema_50': 'Above 50 period moving average' if df['close'].iloc[-1] > df['ema50'].iloc[-1] else False,
            'above_ema_100': 'Above 100 period moving average' if df['close'].iloc[-1] > df['ema100'].iloc[-1] else False,
            'above_ema_200': 'Above 200 period moving average' if df['close'].iloc[-1] > df['ema200'].iloc[-1] else False,
            'below_ema_20': 'Below 20 period moving average' if df['close'].iloc[-1] > df['ema20'].iloc[-1] else False,
            'below_ema_50': 'Below 50 period moving average' if df['close'].iloc[-1] > df['ema50'].iloc[-1] else False,
            'below_ema_100': 'Below 100 period moving average' if df['close'].iloc[-1] > df['ema100'].iloc[-1] else False,
            'below_ema_200': 'Below 200 period moving average' if df['close'].iloc[-1] > df['ema200'].iloc[-1] else False,
        }
        return data, df

    def rsi(self, df):
        df = self.get_rsi(df)
        data = {
            'rsi_oversold': 'RSI is Oversold' if df['rsi'].iloc[-1] <= 30 else False,
            'rsi_overbought': 'RSI is Overbought' if df['rsi'].iloc[-1] >= 70 else False
        }
        return data, df

    def macd_slope(self, ohlc):
        ohlc.ta.macd(close='close', fast=12, slow=26, signal=9, append=True)
        ohlc['macd_slope'] = ohlc['MACD_12_26_9'].rolling(window=2).apply(self.get_slope, raw=True)
        ohlc['macd_sig_slope'] = ohlc['MACDs_12_26_9'].rolling(window=2).apply(self.get_slope, raw=True)
        ohlc['macd_hist_slope'] = ohlc['MACDh_12_26_9'].rolling(window=2).apply(self.get_slope, raw=True)



        data = {'macd_slope': ohlc['macd_slope'].iloc[-1], 'macd_signal': ohlc['MACDs_12_26_9'].iloc[-1], 'macd_hist': ohlc['MACDh_12_26_9'].iloc[-1]}
        data = [
            {'key': 'macd_rising', 'value': 'MACD is rising' if ohlc['macd_slope'].iloc[-1] >= 8 else False, 'data': data},
            {'key': 'macd_dropping', 'value': 'MACD is dropping' if ohlc['macd_slope'].iloc[-1] <= 8 else False, 'data': data},

        ]

        '''
        data = {
            'macd_rising': 'MACD is rising' if ohlc['macd_slope'].iloc[-1] >= 8 else False,
            'macd_dropping': 'MACD is dropping' if ohlc['macd_slope'].iloc[-1] <= 8 else False,
            'cross_soon' : 'MACD Crossover soon' if math.isclose(ohlc['MACD_12_26_9'].iloc[-1], ohlc['MACDs_12_26_9'].iloc[-1], abs_tol=100) else False,
            'macd_over_signal': 'MACD over Signal' if ohlc['MACD_12_26_9'].iloc[-1] > ohlc['MACDs_12_26_9'].iloc[-1] else False,
            'macd_over_centerline': 'MACD over centerline' if ohlc['MACD_12_26_9'].iloc[-1] > 0 else False,
            'macd_under_signal': 'MACD under Signal' if ohlc['MACD_12_26_9'].iloc[-1] < ohlc['MACDs_12_26_9'].iloc[-1] else False,
            'macd_under_centerline': 'MACD under centerline' if ohlc['MACD_12_26_9'].iloc[-1] < 0 else False,
            'macd_signal_crossing_soon' : 'MACD crossover signal soon' if math.isclose(ohlc['MACD_12_26_9'].iloc[-1], ohlc['MACDs_12_26_9'].iloc[-1], abs_tol=100) else False,
        }
        '''
        return data, ohlc

    def bottom_idx(self, df, key, order):
        return argrelextrema(df[key].values, np.less_equal, order=order)[0]

    def top_idx(self, df, key, order):
        return argrelextrema(df[key].values, np.greater_equal, order=order)[0]

    def peak_slope(self, df, key):
        bottom_idx = self.bottom_idx(df, key, 21)
        top_idx = self.top_idx(df, key, 21)
        b = df.iloc[bottom_idx].copy()
        t = df.iloc[top_idx].copy()
        df[f'{key}_low'] = False
        df[f'{key}_low'].loc[bottom_idx] = True
        df[f'{key}_high'] = False
        df[f'{key}_high'].loc[top_idx] = True
        b[f'{key}_lows_slope'] = b[key].rolling(window=4).apply(self.get_slope, raw=True)
        t[f'{key}_highs_slope'] = t[key].rolling(window=4).apply(self.get_slope, raw=True)
        return b[f'{key}_lows_slope'].iloc[-1], t[f'{key}_highs_slope'].iloc[-1], df

    def peaks(self, df):
        close_bottom_slope, close_top_slope, df = self.peak_slope(df, 'close')
        rsi_bottom_slope, rsi_top_slope, df = self.peak_slope(df, 'rsi')
        df.merge(df)
        return close_bottom_slope, close_top_slope, rsi_bottom_slope, rsi_top_slope, df

    def divergence(self, ohlc):
        close_bottom_slope, close_top_slope, rsi_bottom_slope, rsi_top_slope, df = self.peaks(ohlc)
        bullish_regular = (close_bottom_slope < 0) and (rsi_bottom_slope > 0)
        bullish_hidden = (rsi_top_slope < 0) and (close_bottom_slope > 0)
        bearish_regular = (close_top_slope > 0) and (rsi_top_slope < 0)
        bearish_hidden = (close_top_slope < 0) and (rsi_top_slope > 0)


        data = {'close_bottom_slope': close_bottom_slope, 'close_top_slope': close_top_slope, 'rsi_bottom_slope': rsi_bottom_slope, 'rsi_top_slope': rsi_top_slope}
        data = [
            {'key': 'bullish_regular', 'value': 'Bullish divergence' if bullish_regular else False, 'data': data},
            {'key': 'bullish_hidden', 'value': 'Hidden Bullish divergence' if bullish_hidden else False, 'data': data},
            {'key': 'bearish_regular', 'value': 'Bearish divergence' if bearish_regular else False, 'data': data},
            {'key': 'bearish_hidden', 'value': 'Hidden Bearish divergence' if bearish_hidden else False, 'data': data},
        ]

        '''
        data = {
             #'close_bottom_slope': close_bottom_slope,
             #'close_top_slope': close_top_slope,
             #'rsi_bottom_slope': rsi_bottom_slope,
             #'rsi_top_slope': rsi_top_slope,
             'bullish_regular': 'Bullish divergence' if bullish_regular else False,
             'bullish_hidden': 'Hidden Bullish divergence' if bullish_hidden else False,
             'bearish_regular': 'Bearish divergence' if bearish_regular else False,
             'bearish_hidden': 'Hidden Bearish divergence' if bearish_hidden else False,
        }
        '''


        return data, ohlc

    def get_slope(self, array):
        y = np.array(array)
        x = np.arange(len(y))
        slope, intercept, r_value, p_value, std_err = linregress(x,y)
        return slope

    def get_rsi(self, df):
        window_length = 14
        df['diff'] = df['close'].diff(1)
        df['gain'] = df['diff'].clip(lower=0).round(2)
        df['loss'] = df['diff'].clip(upper=0).abs().round(2)
        df['avg_gain'] = df['gain'].rolling(window=window_length, min_periods=window_length).mean()[:window_length+1]
        df['avg_loss'] = df['loss'].rolling(window=window_length, min_periods=window_length).mean()[:window_length+1]
        for i, row in enumerate(df['avg_gain'].iloc[window_length+1:]):
            df['avg_gain'].iloc[i + window_length + 1] = (df['avg_gain'].iloc[i + window_length] * (window_length - 1) + df['gain'].iloc[i + window_length + 1]) / window_length
        for i, row in enumerate(df['avg_loss'].iloc[window_length+1:]):
            df['avg_loss'].iloc[i + window_length + 1] = (df['avg_loss'].iloc[i + window_length] * (window_length - 1) + df['loss'].iloc[i + window_length + 1])  / window_length
        df['rs'] = df['avg_gain'] / df['avg_loss']
        df['rsi'] = 100 - (100 / (1.0 + df['rs']))
        df.drop(['diff', 'gain', 'loss', 'avg_gain', 'avg_gain', 'rs'], axis=1, inplace=True)
        return df