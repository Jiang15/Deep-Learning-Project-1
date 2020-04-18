import torch
import numpy as np
from Project2.Module import Module


class Linear(Module):
    def __init__(self, input_size, output_size):
        super(Linear, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        # 初始化w和b
        std = np.sqrt(2. / (input_size + output_size))
        self.weights = torch.empty(input_size, output_size, dtype=torch.float32).normal(0, std)
        self.bias = torch.empty(output_size, dtype=torch.float32).normal(0, std)
        self.weights_grad = 0
        self.bias_grad = 0

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

class ReLU(Module):
    def forward(self, input):
        self.input = input
        result = input.matmul(input>0)
        return result.float()

    def backward(self, gradwrtoutput):
        gradwrtinput = gradwrtoutput.matmul((self.input>0).float())
        return gradwrtinput
