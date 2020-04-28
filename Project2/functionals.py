import _ini_
import torch


class Relu(object):
    def __init__(self):
        super(Relu, self).__init__()

    def func(self, x):
        return x.clamp(min=0) # Tensor's clamp operator

    def grad(self, x):
        # inter = torch.max(x, torch.zeros_like(x))
        # inter[inter <= 0.] = 0.
        # inter[inter > 0.] = 1.
        out = torch.zeros_like(x)
        out[x>0] = 1
        return out #x * inter


class Sigmoid(object):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def func(self, x):
        return torch.sigmoid(x) #(1.0 / (1 + (-x).exp())).float()

    def grad(self, x):
        return torch.mul(torch.sigmoid(x),1-torch.sigmoid(x))   #((-x).exp() / (1 + (-x).exp().pow(2))).float()


class Tanh(object):
    def __init__(self):
        super(Tanh, self).__init__()

    def func(self, x):
        return torch.tanh(x) #(x.exp() - (-x).exp()) / (x.exp() + (-x).exp()).float()

    def grad(self, x):
        return 1-(torch.tanh(x)).pow(2) #1 / x.cosh().pow(2).float()
