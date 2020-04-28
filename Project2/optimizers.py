import _ini_

class SGD1(object):
    def __init__(self, parameters ,lr):
        self.lr = lr
        self.parameters = parameters
    def step(self):
        for param in self.parameters:
            param.value=param.value-self.lr*param.grad
