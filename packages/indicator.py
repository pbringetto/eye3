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
        return data, df

    def ma(self, df):
        df['ma20'] = df['close'].rolling(20).mean()
        df['ma50'] = df['close'].rolling(50).mean()
        df['ma100'] = df['close'].rolling(100).mean()
        df['ma200'] = df['close'].rolling(200).mean()

        data = {'ma20': df['ma20'].iloc[-1], 'ma50': df['ma50'].iloc[-1], 'ma100': df['ma100'].iloc[-1], 'ma200': df['ma200'].iloc[-1]}
        data = [
            {'key': 'above_ma_20', 'value': 'Above 20 period moving average' if df['close'].iloc[-1] > df['ma20'].iloc[-1] else False, 'data': data},
            {'key': 'above_ma_50', 'value': 'Above 50 period moving average' if df['close'].iloc[-1] > df['ma50'].iloc[-1] else False, 'data': data},
            {'key': 'above_ma_100', 'value': 'Above 100 period moving average' if df['close'].iloc[-1] > df['ma100'].iloc[-1] else False, 'data': data},
            {'key': 'above_ma_200', 'value': 'Above 200 period moving average' if df['close'].iloc[-1] > df['ma200'].iloc[-1] else False, 'data': data},
            {'key': 'below_ma_20', 'value': 'Below 20 period moving average' if df['close'].iloc[-1] > df['ma20'].iloc[-1] else False, 'data': data},
            {'key': 'below_ma_50', 'value': 'Below 50 period moving average' if df['close'].iloc[-1] > df['ma50'].iloc[-1] else False, 'data': data},
            {'key': 'below_ma_100', 'value': 'Below 100 period moving average' if df['close'].iloc[-1] > df['ma100'].iloc[-1] else False, 'data': data},
            {'key': 'below_ma_200', 'value': 'Below 200 period moving average' if df['close'].iloc[-1] > df['ma200'].iloc[-1] else False, 'data': data},
        ]
        return data, df


    def ema(self, df):
        df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema100'] = df['close'].ewm(span=100, adjust=False).mean()
        df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()

        data = {'ema20': df['ema20'].iloc[-1], 'ema50': df['ema50'].iloc[-1], 'ema100': df['ema100'].iloc[-1], 'ema200': df['ema200'].iloc[-1]}
        data = [
            {'key': 'above_ema_20', 'value': 'Above 20 period exponential moving average' if df['close'].iloc[-1] > df['ema20'].iloc[-1] else False, 'data': data},
            {'key': 'above_ema_50', 'value': 'Above 50 period exponential moving average' if df['close'].iloc[-1] > df['ema50'].iloc[-1] else False, 'data': data},
            {'key': 'above_ema_100', 'value': 'Above 100 period exponential moving average' if df['close'].iloc[-1] > df['ema100'].iloc[-1] else False, 'data': data},
            {'key': 'above_ema_200', 'value': 'Above 200 period exponential moving average' if df['close'].iloc[-1] > df['ema200'].iloc[-1] else False, 'data': data},
            {'key': 'below_ema_20', 'value': 'Below 20 period exponential moving average' if df['close'].iloc[-1] > df['ema20'].iloc[-1] else False, 'data': data},
            {'key': 'below_ema_50', 'value': 'Below 50 period exponential moving average' if df['close'].iloc[-1] > df['ema50'].iloc[-1] else False, 'data': data},
            {'key': 'below_ema_100', 'value': 'Below 100 period exponential moving average' if df['close'].iloc[-1] > df['ema100'].iloc[-1] else False, 'data': data},
            {'key': 'below_ema_200', 'value': 'Below 200 period exponential moving average' if df['close'].iloc[-1] > df['ema200'].iloc[-1] else False, 'data': data},
        ]
        return data, df

    def rsi(self, df):
        df = self.get_rsi(df)

        data = {'rsi': df['rsi'].iloc[-1]}
        data = [
            {'key': 'rsi_oversold', 'value': 'RSI is oversold' if df['rsi'].iloc[-1] <= 30 and df['rsi'].iloc[-1] >= 20 else False, 'data': data},
            {'key': 'rsi_extremely_oversold', 'value': 'RSI is extremely oversold' if df['rsi'].iloc[-1] <= 20 else False, 'data': data},
            {'key': 'rsi_overbought', 'value': 'RSI is overbought' if df['rsi'].iloc[-1] >= 70  and df['rsi'].iloc[-1] <= 80 else False, 'data': data},
            {'key': 'rsi_extremely_overbought', 'value': 'RSI is extremely overbought' if df['rsi'].iloc[-1] >= 80 else False, 'data': data},
        ]
        return data, df

    def macd_slope(self, df):
        df.ta.macd(close='close', fast=12, slow=26, signal=9, append=True)
        df['macd_slope'] = df['MACD_12_26_9'].rolling(window=2).apply(self.get_slope, raw=True)
        df['macd_sig_slope'] = df['MACDs_12_26_9'].rolling(window=2).apply(self.get_slope, raw=True)
        df['macd_hist_slope'] = df['MACDh_12_26_9'].rolling(window=2).apply(self.get_slope, raw=True)
        data = {'macd_slope': df['macd_slope'].iloc[-1], 'macd_signal': df['MACDs_12_26_9'].iloc[-1], 'macd_hist': df['MACDh_12_26_9'].iloc[-1]}
        data = [
            {'key': 'macd_rising', 'value': 'MACD is rising' if df['macd_slope'].iloc[-1] >= 8 else False, 'data': data},
            {'key': 'macd_dropping', 'value': 'MACD is dropping' if df['macd_slope'].iloc[-1] <= 8 else False, 'data': data},
            {'key': 'cross_soon', 'value': 'MACD Crossover soon' if math.isclose(df['MACD_12_26_9'].iloc[-1], df['MACDs_12_26_9'].iloc[-1], abs_tol=100) else False, 'data': data},
            {'key': 'macd_over_signal', 'value': 'MACD over Signal' if df['MACD_12_26_9'].iloc[-1] > df['MACDs_12_26_9'].iloc[-1] else False, 'data': data},
            {'key': 'macd_over_centerline', 'value': 'MACD over centerline' if df['MACD_12_26_9'].iloc[-1] > 0 else False, 'data': data},
            {'key': 'macd_under_signal', 'value': 'MACD under Signal' if df['MACD_12_26_9'].iloc[-1] < df['MACDs_12_26_9'].iloc[-1] else False, 'data': data},
            {'key': 'macd_under_centerline', 'value': 'MACD under centerline' if df['MACD_12_26_9'].iloc[-1] < 0 else False, 'data': data},
            {'key': 'macd_signal_crossing_soon', 'value': 'MACD crossover signal soon' if math.isclose(df['MACD_12_26_9'].iloc[-1], df['MACDs_12_26_9'].iloc[-1], abs_tol=100) else False, 'data': data},
        ]
        return data, df

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


    def falling_wedge(self, df):
          key = 'close'
          b, t = df.iloc[self.bottom_idx(df, key, 5)].copy(), df.iloc[self.top_idx(df, key, 2)].copy()
          b[f'{key}_lows_slope'] = b[key].rolling(window=5).apply(self.get_slope, raw=True)
          t[f'{key}_highs_slope'] = t[key].rolling(window=2).apply(self.get_slope, raw=True)

          pattern = False
          if b[f'{key}_lows_slope'].iloc[-1] < 0:
              pattern = True
          if t[f'{key}_highs_slope'].iloc[-1] < 0:
              pattern = True
          if t[f'{key}_highs_slope'].iloc[-1] > b[f'{key}_lows_slope'].iloc[-1]:
              pattern = True
          i = 3
          while i > 0:
              if (t['close'].iloc[-i] - b['close'].iloc[-i]) < (t['close'].iloc[-(i-1)] - b['close'].iloc[-(i-1)]):
                  pattern = True
              i -= 1

          data = [
            {'key': 'falling_wedge', 'value': 'Falling wedge' if pattern else False, 'data': 0},
          ]
          print(pattern)
          return data


    def divergence(self, df):
        close_bottom_slope, close_top_slope, rsi_bottom_slope, rsi_top_slope, df = self.peaks(df)
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