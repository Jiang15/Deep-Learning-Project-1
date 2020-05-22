import torch

from Module import Module

class MSELoss(Module):
    def __init__(self):
        super(MSELoss,self).__init__()
        self.input=0
    def forward(self,output, target):
        output = output.view(target.size())
        loss= ((output-target)**2).mean()
        return loss

    def backward(self,output,target):
        output = output.view(target.size())
        grad=2*(output-target)/output.numel()
        return grad
