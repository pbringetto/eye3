import models.model as m
import json

class SignalDataModel:
    def __init__(self):
        self.model = m.Model()

    def get_signals(self, pair, timeframe):
        sql = """ SELECT * FROM `signal`
                  WHERE timeframe = %s AND pair = %s
                  ORDER BY id DESC LIMIT 100 """
        params = (timeframe, pair, )
        return self.model.select_all(sql, params)

    def get_all_signals(self, limit):
        sql = """ SELECT * FROM `signal`
                  ORDER BY id DESC LIMIT %s """
        params = (limit, )
        return self.model.select_all(sql, params)

    def get_signal(self, pair, timeframe, key):
        sql = """ SELECT * FROM `signal`
                  WHERE timeframe = %s AND pair = %s AND `key` = %s
                  ORDER BY id DESC LIMIT 1 """
        params = (timeframe, pair, key, )
        return self.model.select_one(sql, params)

    def insert_signal(self, pair, timeframe, key, value, ohlc_timestamp, data_id):
        sql = "INSERT IGNORE INTO `signal` (pair, timeframe, `key`, `value`, ohlc_timestamp, data_id) VALUES (%s, %s, %s, %s, %s, %s)"
        params = (pair, timeframe, key, value, ohlc_timestamp, data_id, )
        return self.model.insert(sql, params)

    def insert_data(self, ohlc_timestamp, open, high, low, close, volume, ma20, ma50, ma100, ma200, ema20, ema50, ema100, ema200, std, bollinger_high, bollinger_low, rsi, macd, macdh, macds, macd_slope, macd_sig_slope, macd_hist_slope):
        sql = "INSERT IGNORE INTO `data` (`ohlc_timestamp`, `open`, `high`, `low`, `close`, `volume`, `ma20`, `ma50`, `ma100`, `ma200`, `ema20`, `ema50`, `ema100`, `ema200`, `std`, `bollinger_high`, `bollinger_low`, `rsi`, `macd`, `macdh`, `macds`, `macd_slope`, `macd_sig_slope`, `macd_hist_slope`) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        params = (ohlc_timestamp, open, high, low, close, volume, ma20, ma50, ma100, ma200, ema20, ema50, ema100, ema200, std, bollinger_high, bollinger_low, rsi, macd, macdh, macds, macd_slope, macd_sig_slope, macd_hist_slope, )
        return self.model.insert(sql, params)

    def get_data(self, id):
        sql = " SELECT * FROM `data` WHERE id = %s "
        params = (id, )
        return self.model.select_one(sql, params)
