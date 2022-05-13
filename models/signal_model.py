import models.model as m

class SignalDataModel:
    def __init__(self):
        self.model = m.Model()

    def get_signals(self, pair, timeframe):
        sql = """ SELECT * FROM `signal`
                  WHERE timeframe = %s AND pair = %s
                  ORDER BY id DESC LIMIT 100"""
        params = (timeframe, pair, )
        return self.model.select_all(sql, params)

    def insert_signal(self, pair, timeframe, key, value):
        sql = "INSERT IGNORE INTO `signal` (pair, timeframe, `key`, `value`) VALUES (%s, %s, %s, %s)"
        params = (pair, timeframe, key, value, )
        return self.model.insert(sql, params)