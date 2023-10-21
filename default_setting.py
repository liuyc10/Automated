class DefaultSetting(object):

    def __init__(self):

        self.DEBUG = False

        self.fa = 3
        self.gauss = 3

        self.th1sr = 80
        self.th2sr = 200

        self.th1cr = 140
        self.th2cr = 250

        self.th1ca = 200
        self.th2ca = 250

        self.scale = 2

    def data_set(self):
        return [self.fa, self.gauss, self.th1sr, self.th2sr, self.th1cr, self.th2cr, self.th1ca, self.th2ca, self.scale]
