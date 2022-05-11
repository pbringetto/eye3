import api.ftx as f
import cfg_load
alpha = cfg_load.load('/home/ubuntu/eye3/alpha.yaml')

def go()
    ftx = f.FtxClient(alpha["key"], alpha["secret"])

    for pair in alpha["pairs"]:
         return ftx.get_historical_prices(pair['pair'], 86400)





print(go())


