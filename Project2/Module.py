class Module(object):
    def _init_(self):
        self.module = 0

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *output):
        raise NotImplementedError

    def param(self):
        return []

    def zero_grad(self):
        pass
