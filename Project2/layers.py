# from distributed.protocol.tests.test_torch import torch
import torch
from Project2.Module import Module, Parameters
import numpy as np


class Linear(Module):
    def __init__(self, in_nodes, out_nodes):
        super(Linear, self).__init__()
        type = torch.float32
        std = np.sqrt(2. / (in_nodes + out_nodes))
        self.weights = Parameters(torch.zeros(out_nodes, in_nodes, dtype=torch.float32).normal_(0, std))
        self.bias = Parameters(torch.zeros(out_nodes, dtype=torch.float32))
        self.result = Parameters(torch.zeros(out_nodes, dtype=torch.float32))
        self.input = Parameters(torch.zeros(in_nodes, dtype=torch.float32))

    def zero_grad(self):
        self.weights.grad.zero_()
        self.bias.grad.zero_()

    def forward(self, x):
        self.input.value = x
        self.result.value = x.matmul(self.weights.value.t()) + self.bias.value
        return self.result.value

    def backward(self, grad):
        self.weights.grad = self.weights.grad + torch.mm(grad.t(), self.input.value)
        self.bias.grad = self.bias.grad + grad.sum(0)
        grad_t = torch.mm(grad, self.weights.value)

        return grad_t

    def param(self):
        return self.weights, self.bias


class Relu(Module):
    def __init__(self):
        super(Relu, self).__init__()
        self.input = torch.empty(1)

    def forward(self, x):
        self.input = x
        return x.clamp(min=0)

    # def backward(self,x):
    # out=self.input
    # out[out > 0.] = 1.
    # return out*x
    def backward(self, gradwrtoutput):
        '''
        ReLU backward:
        The gradient of ReLU is 0 if input<0 else 1
        :param gradwrtoutput: dL/d(output) Tensor with the same shape as input
        :return: dL/d(input): Tensor with the same shape as input
        '''
        grad = torch.empty(*gradwrtoutput.shape)
        grad[self.input > 0] = 1
        grad[self.input <= 0] = 0
        return grad * gradwrtoutput


class Leaky_Relu(Module):
    def __init__(self):
        super(Leaky_Relu, self).__init__()
        self.input = torch.empty(1)
    def forward(self, input):
        self.input = input
        return input * (input>0).float().add(0.01 * input * (input<0).float())

    def backward(self, gradwrtoutput):
        grad = torch.empty(*gradwrtoutput.shape)
        grad[self.input > 0] = 1
        grad[self.input <= 0] = 0.01
        return grad * gradwrtoutput

class Elu(Module):
    def __init__(self):
        super(Elu, self).__init__()
        self.input = torch.empty(1)
    def forward(self, input):
        self.input = input
        return input * (input>0).float().add(0.01 *(torch.exp(input) -1) * (input<=0).float())

    def backward(self, gradwrtoutput):
        grad = 0.01*torch.exp(self.input)* (self.input<=0).float().add((self.input>0).float())
        return grad * gradwrtoutput

class Sigmoid(Module):
    def __init__(self):
        super(Sigmoid, self).__init__()
        self.input = torch.empty(1)

    def forward(self, input):
        self.input = input

        return input.sigmoid()
    # Backward pass
    def backward(self, gradwrtoutput):
        derivatives = torch.mul(self.input.sigmoid(), 1 - self.input.sigmoid())
        return derivatives * gradwrtoutput


class Tanh(Module):

    def __init__(self):
        self.input = None

    def forward(self, input):
        self.input = input
        return input.tanh()

    def backward(self, gradwrtoutput):
        grad = 1 - (self.input.tanh()) ** 2
        return grad * gradwrtoutput
