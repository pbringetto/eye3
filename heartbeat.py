import api.ftx as f
import strategy as s
import cfg_load
import twitter as t
alpha = cfg_load.load('alpha.yaml')
import pandas as pd
import helpers.util as u
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)

def go():
    strategy = s.Strategy()
    twitter = t.Twitter()
    ftx = f.FtxClient(alpha["ftx_key"], alpha["ftx_secret"])
    tf = 86400
    for pair in alpha["pairs"]:
        for tf in alpha["timeframes"]:

             df = pd.DataFrame(ftx.get_historical_prices(pair['pair'], tf['seconds']))

             data, df = strategy.setup(df, tf['seconds'], pair['pair'])

             u.show('Market', pair['label'])
             u.show('Timeframe', tf['label'])
             u.show('Open Price', df['open'].iloc[-1])
             u.show('High', df['high'].iloc[-1])
             u.show('Low', df['low'].iloc[-1])

             for key, value in data.items():
                 if value:
                     print(value)

             twitter.tweet()

go()


