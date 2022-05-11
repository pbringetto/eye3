import api.ftx as f
import cfg_load
alpha = cfg_load.load('/home/ubuntu/eye3/alpha.yaml')
import pandas as pd
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)

def go():
    ftx = f.FtxClient(alpha["key"], alpha["secret"])

    for pair in alpha["pairs"]:
         df = pd.DataFrame(ftx.get_historical_prices(pair['pair'], 86400))
         print(df)
go()


