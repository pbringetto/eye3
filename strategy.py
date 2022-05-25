import packages.indicator as i

class Strategy:
    def __init__(self):
        self.indicator = i.Indicator()

    def setup(self, ohlc, tf, pair):

        price = float(ohlc['close'][::-1][0])

        ma_data, ohlc = self.indicator.ma(ohlc)
        ema_data, ohlc = self.indicator.ema(ohlc)
        bollinger_data, ohlc = self.indicator.bollinger(ohlc)

        rsi_data, ohlc = self.indicator.rsi(ohlc)
        macd_data, ohlc = self.indicator.macd_slope(ohlc)
        div_data, ohlc = self.indicator.divergence(ohlc)

        obv_data, ohlc = self.indicator.on_balance_volume(ohlc, 5)

        #print(ohlc[['close', 'volume', 'bollinger_low', 'bollinger_high']].iloc[-20:])

        data = div_data + bollinger_data + macd_data + ma_data + ema_data

        return data, ohlc
