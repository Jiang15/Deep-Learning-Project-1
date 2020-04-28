import torch
import numpy as np
from Project2.Module import Module


class Linear(Module):
    def __init__(self, input_size, output_size):
        super(Linear, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.weights_grad = torch.zeros([self.input_size,self.output_size])
        self.bias_grad = torch.zeros([self.output_size])
        self.weights = torch.empty([input_size,output_size])
        self.bias = torch.empty([output_size])

    def forward(self, input):
        self.input = input
        result = input.matmul(self.weights).add(self.bias)
        return result

    def backward(self, gradwrtoutput):
        gradwrtinput = self.weights.t().matmul(gradwrtoutput)
        self.weights_grad.add(gradwrtinput.matmul(self.input))
        self.bias_grad.add(gradwrtinput)
        return gradwrtinput

    def param(self):
        return [self.weights, self.weights_grad, self.bias, self.bias_grad]

    def zero_grad(self):
        self.weights_grad = torch.zeros([self.input_size,self.output_size])
        self.bias_grad = torch.zeros([self.output_size])

class ReLU(Module):
    def __init__(self):
        super(ReLU,self).__init__()
    def forward(self, input):
        self.input = input
        input[input<0]= 0.
        return input

    def backward(self, gradwrtoutput):
        out = self.input
        out[out>0.] = 1.
        out[out<0.] = 0.
        gradwrtinput = gradwrtoutput.matmul(out)
        return gradwrtinput
