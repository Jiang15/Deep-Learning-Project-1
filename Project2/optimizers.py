import _ini_
import torch

class Adam(object):
    def __init__(self, parameters, lr):
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.lr = lr
        self.epsilon = 1e-8
        self.iter = 1
        self.parameters = parameters
        self.m = [torch.zeros(param.grad.size()) for param in self.parameters]
        self.v = [torch.zeros(param.grad.size()) for param in self.parameters]

    def update(self):
        for i, param in enumerate(self.parameters):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * param.grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * param.grad * param.grad
            m_hat = self.m[i] / (1 - torch.pow(self.beta1, torch.FloatTensor([self.iter + 1])))
            v_hat = self.v[i] / (1 - torch.pow(self.beta2, torch.FloatTensor([self.iter + 1])))
            param.value = param.value - self.lr * m_hat / (torch.sqrt(v_hat) + self.epsilon)

        self.iter = self.iter + 1
