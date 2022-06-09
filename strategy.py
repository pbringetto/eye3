import ta_signals
import pyangles

class Strategy:
    def setup(self, ohlc, tf, pair, window, windows, order):
        data, ohlc = ta_signals.go(ohlc, 'close', window)
        pattern_data, lows, highs = pyangles.go(ohlc, 'close', windows, order)
        return data, pattern_data, ohlc, lows, highs
