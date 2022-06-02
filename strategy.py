import ta_signals
import pyangles

class Strategy:
    def setup(self, ohlc, tf, pair):
        data = ta_signals.go(ohlc, 'close')
        pattern_data = pyangles.go(ohlc, 'close', [4, 4], [1, 1])
        return data + pattern_data, ohlc
