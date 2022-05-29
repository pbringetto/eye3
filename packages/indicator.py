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
            {'key': 'below_bollinger_low', 'value': 'Below bollinger Low' if df['close'].iloc[-1] < df['bollinger_low'].iloc[-1] else False},
            {'key': 'above_bollinger_high', 'value': 'Above bollinger high' if df['close'].iloc[-1] > df['bollinger_high'].iloc[-1] else False},
            {'key': 'at_bollinger_low', 'value': 'At Bollinger low' if math.isclose(df['close'].iloc[-1], df['bollinger_low'].iloc[-1], abs_tol=100) else False},
            {'key': 'at_bollinger_high', 'value': 'At Bollinger high' if math.isclose(df['close'].iloc[-1], df['bollinger_high'].iloc[-1], abs_tol=100) else False},
        ]
        return data, df

    def ma(self, df):
        df['ma20'] = df['close'].rolling(20).mean()
        df['ma50'] = df['close'].rolling(50).mean()
        df['ma100'] = df['close'].rolling(100).mean()
        df['ma200'] = df['close'].rolling(200).mean()

        data = {'ma20': df['ma20'].iloc[-1], 'ma50': df['ma50'].iloc[-1], 'ma100': df['ma100'].iloc[-1], 'ma200': df['ma200'].iloc[-1]}
        data = [
            {'key': 'above_ma_20', 'value': 'Above 20 period moving average' if df['close'].iloc[-1] > df['ma20'].iloc[-1] else False},
            {'key': 'above_ma_50', 'value': 'Above 50 period moving average' if df['close'].iloc[-1] > df['ma50'].iloc[-1] else False},
            {'key': 'above_ma_100', 'value': 'Above 100 period moving average' if df['close'].iloc[-1] > df['ma100'].iloc[-1] else False},
            {'key': 'above_ma_200', 'value': 'Above 200 period moving average' if df['close'].iloc[-1] > df['ma200'].iloc[-1] else False},
            {'key': 'below_ma_20', 'value': 'Below 20 period moving average' if df['close'].iloc[-1] < df['ma20'].iloc[-1] else False},
            {'key': 'below_ma_50', 'value': 'Below 50 period moving average' if df['close'].iloc[-1] < df['ma50'].iloc[-1] else False},
            {'key': 'below_ma_100', 'value': 'Below 100 period moving average' if df['close'].iloc[-1] < df['ma100'].iloc[-1] else False},
            {'key': 'below_ma_200', 'value': 'Below 200 period moving average' if df['close'].iloc[-1] < df['ma200'].iloc[-1] else False},
        ]
        return data, df


    def ema(self, df):
        df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema100'] = df['close'].ewm(span=100, adjust=False).mean()
        df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()

        data = {'ema20': df['ema20'].iloc[-1], 'ema50': df['ema50'].iloc[-1], 'ema100': df['ema100'].iloc[-1], 'ema200': df['ema200'].iloc[-1]}
        data = [
            {'key': 'above_ema_20', 'value': 'Above 20 period exponential moving average' if df['close'].iloc[-1] > df['ema20'].iloc[-1] else False},
            {'key': 'above_ema_50', 'value': 'Above 50 period exponential moving average' if df['close'].iloc[-1] > df['ema50'].iloc[-1] else False},
            {'key': 'above_ema_100', 'value': 'Above 100 period exponential moving average' if df['close'].iloc[-1] > df['ema100'].iloc[-1] else False},
            {'key': 'above_ema_200', 'value': 'Above 200 period exponential moving average' if df['close'].iloc[-1] > df['ema200'].iloc[-1] else False},
            {'key': 'below_ema_20', 'value': 'Below 20 period exponential moving average' if df['close'].iloc[-1] > df['ema20'].iloc[-1] else False},
            {'key': 'below_ema_50', 'value': 'Below 50 period exponential moving average' if df['close'].iloc[-1] > df['ema50'].iloc[-1] else False},
            {'key': 'below_ema_100', 'value': 'Below 100 period exponential moving average' if df['close'].iloc[-1] > df['ema100'].iloc[-1] else False},
            {'key': 'below_ema_200', 'value': 'Below 200 period exponential moving average' if df['close'].iloc[-1] > df['ema200'].iloc[-1] else False},
        ]
        return data, df

    def rsi(self, df):
        df = self.get_rsi(df)

        data = {'rsi': df['rsi'].iloc[-1]}
        data = [
            {'key': 'rsi_oversold', 'value': 'RSI is oversold' if df['rsi'].iloc[-1] <= 30 and df['rsi'].iloc[-1] >= 20 else False},
            {'key': 'rsi_extremely_oversold', 'value': 'RSI is extremely oversold' if df['rsi'].iloc[-1] <= 20 else False},
            {'key': 'rsi_overbought', 'value': 'RSI is overbought' if df['rsi'].iloc[-1] >= 70  and df['rsi'].iloc[-1] <= 80 else False},
            {'key': 'rsi_extremely_overbought', 'value': 'RSI is extremely overbought' if df['rsi'].iloc[-1] >= 80 else False},
        ]
        return data, df

    def macd_slope(self, df):
        df.ta.macd(close='close', fast=12, slow=26, signal=9, append=True)
        df['macd_slope'] = df['MACD_12_26_9'].rolling(window=2).apply(self.get_slope, raw=True)
        df['macd_sig_slope'] = df['MACDs_12_26_9'].rolling(window=2).apply(self.get_slope, raw=True)
        df['macd_hist_slope'] = df['MACDh_12_26_9'].rolling(window=2).apply(self.get_slope, raw=True)
        data = {'macd_slope': df['macd_slope'].iloc[-1], 'macd_signal': df['MACDs_12_26_9'].iloc[-1], 'macd_hist': df['MACDh_12_26_9'].iloc[-1]}
        bullish_cross = math.isclose(df['MACD_12_26_9'].iloc[-1], df['MACDs_12_26_9'].iloc[-1], abs_tol=100) and df['macd_slope'].iloc[-1] > 0
        bearish_cross = math.isclose(df['MACD_12_26_9'].iloc[-1], df['MACDs_12_26_9'].iloc[-1], abs_tol=100) and df['macd_slope'].iloc[-1] < 0
        data = [
            {'key': 'macd_rising', 'value': 'MACD is rising' if df['macd_slope'].iloc[-1] >= 8 else False},
            {'key': 'macd_dropping', 'value': 'MACD is dropping' if df['macd_slope'].iloc[-1] <= 8 else False},
            {'key': 'bullish_macd_cross', 'value': 'Bullish MACD Crossover soon' if bullish_cross else False},
            {'key': 'bearish_macd_cross', 'value': 'Bearish MACD Crossover soon' if bearish_cross else False},
            {'key': 'macd_over_signal', 'value': 'MACD over Signal' if df['MACD_12_26_9'].iloc[-1] > df['MACDs_12_26_9'].iloc[-1] else False},
            {'key': 'macd_over_centerline', 'value': 'MACD over centerline' if df['MACD_12_26_9'].iloc[-1] > 0 else False},
            {'key': 'macd_under_signal', 'value': 'MACD under Signal' if df['MACD_12_26_9'].iloc[-1] < df['MACDs_12_26_9'].iloc[-1] else False},
            {'key': 'macd_under_centerline', 'value': 'MACD under centerline' if df['MACD_12_26_9'].iloc[-1] < 0 else False},
            {'key': 'macd_signal_crossing_soon', 'value': 'MACD crossover signal soon' if math.isclose(df['MACD_12_26_9'].iloc[-1], df['MACDs_12_26_9'].iloc[-1], abs_tol=100) else False},
        ]
        return data, df

    def low_idx(self, df, key, order):
        return argrelextrema(df[key].values, np.less_equal, order=order)[0]

    def high_idx(self, df, key, order):
        return argrelextrema(df[key].values, np.greater_equal, order=order)[0]

    def peak_slope(self, df, key):
        lows, highs = self.range_slopes(df, key, 21, 21)
        lows[f'{key}_lows_slope'] = lows[key].rolling(window=4).apply(self.get_slope, raw=True)
        highs[f'{key}_highs_slope'] = highs[key].rolling(window=4).apply(self.get_slope, raw=True)
        return lows[f'{key}_lows_slope'].iloc[-1], highs[f'{key}_highs_slope'].iloc[-1], df

    def rsi_close_div_peak_slopes(self, df):
        close_low_slope, close_high_slope, df = self.peak_slope(df, 'close')
        rsi_low_slope, rsi_high_slope, df = self.peak_slope(df, 'rsi')
        df.merge(df)
        return close_low_slope, close_high_slope, rsi_low_slope, rsi_high_slope, df

    def high_low_idx(self, df, key, low_window, high_window):
        return self.low_idx(df, key, low_window), self.high_idx(df, key, high_window)

    def range_less_than_prev(self, h, l, prev_h, prev_l):
        print('------------------------')
        print('prev high:' + str(prev_h))
        print('prev low:' + str(prev_l))
        print('high:' + str(h))
        print('low:' + str(l))
        print((h - l) < (prev_h - prev_l))
        print('------------------------')
        return (h - l) < (prev_h - prev_l)

    def is_compressing(self, highs, lows, i = 3):
         r = False
         while i > 0:
             if self.range_less_than_prev(highs['close'].iloc[-i], lows['close'].iloc[-i], highs['close'].iloc[-(i-1)], lows['close'].iloc[-(i-1)]):
                 r = True
             #else:
             #    r = False
             #    break
             i -= 1
         return r

    def range_slopes(self, df, key, low_window, high_window):
        low_idx, high_idx = self.high_low_idx(df, key, low_window, high_window)
        lows, highs = df.iloc[low_idx].copy(), df.iloc[high_idx].copy()
        lows[f'{key}_lows_slope'] = lows[key].rolling(window=5).apply(self.get_slope, raw=True)
        highs[f'{key}_highs_slope'] = highs[key].rolling(window=2).apply(self.get_slope, raw=True)
        return lows, highs

    def falling_wedge(self, df, key, tf):
        df.drop(df.tail(2).index,inplace=True)
        lows, highs = self.range_slopes(df, key, tf['slope_window'][0], tf['slope_window'][1])

        r = False
        if lows[f'{key}_lows_slope'].iloc[-1] < 0:
            r = True
        if highs[f'{key}_highs_slope'].iloc[-1] < 0:
            r = True
        if highs[f'{key}_highs_slope'].iloc[-1] > lows[f'{key}_lows_slope'].iloc[-1]:
            r = True
        return r

    def patterns(self, df, tf):
          key = 'close'
          falling_wedge = self.falling_wedge(df, key, tf)
          data = [
            {'key': 'falling_wedge', 'value': 'Falling wedge' if falling_wedge else False},
          ]
          return data

    def divergence(self, df):
        close_low_slope, close_high_slope, rsi_low_slope, rsi_high_slope, df = self.rsi_close_div_peak_slopes(df)
        bullish_regular = (close_low_slope < 0) and (rsi_low_slope > 0)
        bullish_hidden = (rsi_high_slope < 0) and (close_low_slope > 0)
        bearish_regular = (close_high_slope > 0) and (rsi_high_slope < 0)
        bearish_hidden = (close_high_slope < 0) and (rsi_high_slope > 0)
        data = {'close_low_slope': close_low_slope, 'close_high_slope': close_high_slope, 'rsi_low_slope': rsi_low_slope, 'rsi_high_slope': rsi_high_slope}
        data = [
            {'key': 'bullish_regular', 'value': 'Bullish divergence' if bullish_regular else False},
            {'key': 'bullish_hidden', 'value': 'Hidden Bullish divergence' if bullish_hidden else False},
            {'key': 'bearish_regular', 'value': 'Bearish divergence' if bearish_regular else False},
            {'key': 'bearish_hidden', 'value': 'Hidden Bearish divergence' if bearish_hidden else False},
        ]
        return data, df

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