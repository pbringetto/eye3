import models.model as m

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

    def insert_signal(self, pair, timeframe, key, value, ohlc_timestamp, price):
        sql = "INSERT IGNORE INTO `signal` (pair, timeframe, `key`, `value`, ohlc_timestamp, price) VALUES (%s, %s, %s, %s, %s, %s)"
        params = (pair, timeframe, key, value, ohlc_timestamp, price, )
        return self.model.insert(sql, params)