import packages.indicator as i

class Strategy:
    def __init__(self):
        self.indicator = i.Indicator()

    def setup(self, ohlc, tf, pair):
        price = float(ohlc['close'][::-1][0])

        ma_data, ohlc = self.indicator.ma(ohlc, tf)
        ema_data, ohlc = self.indicator.ema(ohlc, tf)
        bollinger_data, ohlc = self.indicator.bollinger(ohlc, tf)

        rsi_data, ohlc = self.indicator.rsi(ohlc, tf)
        macd_data, ohlc = self.indicator.macd_slope(ohlc, tf)
        div_data, ohlc = self.indicator.divergence(ohlc, tf)

        data = {**div_data, **macd_data, **rsi_data, **ma_data, **ema_data, **bollinger_data}

        return data, ohlc
