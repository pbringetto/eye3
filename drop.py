import data as d

class Drop:
    def __init__(self):
        self.data = d.Data()

    def history(self):
        return self.data.get_signals(10)

    def alpha(self):
        #macd_rising_following_oversold_divergence
        data = self.data.signals()
        data = self.directions(data)
        return data

    def directions(self, data):
        for pk,pv  in data.items():
            print(pk)
            for tk,tv  in pv.items():
                print(tk)
                for sk, sv  in tv.items():
                    print(sk)
                    for ck, cv  in sv.items():
                        print(ck)
                        print(cv)
                        for item in cv:
                            print(item['key'])
        return data