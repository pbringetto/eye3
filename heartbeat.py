import api.ftx as f
import strategy as s
import cfg_load
alpha = cfg_load.load('/home/ubuntu/eye3/alpha.yaml')
import pandas as pd
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)

def go():
    strategy = s.Strategy()
    ftx = f.FtxClient(alpha["key"], alpha["secret"])
    tf = 86400
    for pair in alpha["pairs"]:
         df = pd.DataFrame(ftx.get_historical_prices(pair['pair'], tf))
         setup(df, tf, pair['pair'])
go()


