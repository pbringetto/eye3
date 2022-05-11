import pandas as pd
import pandas_ta as ta
import numpy as np
import cfg_load
import helpers.util as u
alpha = cfg_load.load('/home/ubuntu/eye3/alpha.yaml')
pd.set_option('display.max_rows', 2000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)

class Strategy:
    def __init__(self):

    def bottom_idx(self, df, key, order):
        return argrelextrema(df[key].values, np.less_equal, order=order)[0]

    def top_idx(self, df, key, order):
        return argrelextrema(df[key].values, np.greater_equal, order=order)[0]

    def peak_slope(self, df, tf, key):
        bottom_idx = self.bottom_idx(df, key, tf['extrema_order'])
        top_idx = self.top_idx(df, key, tf['extrema_order'])
        b = df.iloc[bottom_idx].copy()
        t = df.iloc[top_idx].copy()

        b[f'{key}_lows_slope'] = b[key].rolling(window=tf["extrema_peaks"]).apply(self.get_slope, raw=True)
        t[f'{key}_high_slope'] = t[key].rolling(window=tf["extrema_peaks"]).apply(self.get_slope, raw=True)

        return b[f'{key}_lows_slope'].iloc[-1], t[f'{key}_high_slope'].iloc[-1]

    def divergence(self, df, tf):
        close_bottom_slope, close_top_slope = self.peak_slope(df, tf, 'close')
        rsi_bottom_slope, rsi_top_slope = self.peak_slope(df, tf, 'rsi')

        print('divergence------------------------')
        u.show('close_bottom_slope', close_bottom_slope)
        u.show('close_top_slope', close_top_slope)
        u.show('rsi_bottom_slope', rsi_bottom_slope)
        u.show('rsi_top_slope', rsi_top_slope)

        bullish_regular = (close_bottom_slope < 0) and (rsi_bottom_slope > 0)
        bullish_hidden = (rsi_top_slope < 0) and (close_bottom_slope > 0)
        bullish_exaggerated = (math.isclose(0, close_bottom_slope) and rsi_bottom_slope > 0) or (math.isclose(0, rsi_bottom_slope) and close_bottom_slope < 0)

        bearish_regular = (close_top_slope > 0) and (rsi_top_slope < 0)
        bearish_hidden = (close_top_slope < 0) and (rsi_top_slope > 0)
        bearish_exaggerated = (math.isclose(0, close_top_slope) and rsi_top_slope < 0) or (math.isclose(0, rsi_top_slope) and close_top_slope > 0)

        u.show('bullish_regular', bullish_regular)
        u.show('bullish_hidden', bullish_hidden)
        u.show('bearish_regular', bearish_regular)
        u.show('bearish_hidden', bearish_hidden)
        u.show('bullish_exaggerated', bearish_regular)
        u.show('bearish_exaggerated', bearish_hidden)

    def setup(self, ohlc, tf, pair):
        price = float(ohlc['close'][::-1][0])
        ohlc = self.indicator.rsi(ohlc)

        self.divergence(ohlc, tf)
        u.show_object('strategy data', ohlc.iloc[-1])

    def get_slope(self, array):
        y = np.array(array)
        x = np.arange(len(y))
        slope, intercept, r_value, p_value, std_err = linregress(x,y)
        return slope