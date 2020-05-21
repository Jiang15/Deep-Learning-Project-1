import _ini_
import torch


class SGD(object):
    def __init__(self, parameters, lr):
        self.lr = lr
        self.parameters = parameters

    def update(self):
        for param in self.parameters:
            param.value = param.value - self.lr * param.grad


class MomentumSGD(Module):
    def __init__(self, parameters ,lr, rho):
        self.lr= lr
        self.rho = rho
        self.parameters = parameters
        self.r=[]
        for param in parameters:
            self.r.append(torch.zeros_like(param.value))
    def update(self):
        for param,r in zip(self.parameters,self.r):
            r = self.rho * r + param.grad
            param.value =param.value- self.lr* r


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


class AdaGrad(Module):
    def __init__(self, parameters ,lr,delta):
        self.lr = lr
        self.delta=delta
        self.parameters = parameters
        self.r=[]
        for param in parameters:
            self.r.append(torch.zeros_like(param.value))
    def step(self):
        for i, p in zip(range(len(self.r)),self.parameters):
            self.r[i] = self.r[i]+ torch.mul(p.grad, p.grad)
            p.value =p.value-self.lr * p.grad /(self.delta + torch.sqrt(self.r[i]))

