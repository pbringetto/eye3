import api.ftx a f
import cfg_load
alpha = cfg_load.load('/home/ubuntu/eye3/alpha.yaml')

def go(x)
    ftx = f.FtxClient(alpha["key"], alpha["secret"])
    print(x)

go('hello')


