import _ini_


class SGD(object):
    def __init__(self, parameters ,lr):
        self.lr = lr
        self.parameters = parameters
    def step(self):
        self.parameters[:,0] += self.parameters[:,1] * self.lr
        self.parameters[:,2] += self.parameters[:,4] * self.lr
