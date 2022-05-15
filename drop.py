import data as d

class Drop:
    def __init__(self):
        self.data = d.Data()

    def alpha(self):
        print(self.data.signals())