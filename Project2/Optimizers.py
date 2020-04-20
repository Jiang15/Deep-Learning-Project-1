import _ini_
class SGD(object):
    def __init__(self, learning_rate):
        super(SGD, self).__init__()
        self.learning_rate = learning_rate

    def step(self, model):
        for k, layer in enumerate(model.layers):
            layer.weights = layer.weights - self.learning_rate * layer.weights.grad
            layer.bias= layer.bias - self.learning_rate * layer.bias.grad

    def zero_grad(self, model):
        self.weights.grad.zero_()
        self.bias.grad.zero_()


