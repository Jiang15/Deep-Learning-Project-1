import torch

class SGD(object):
    def __init__(self, parameters, lr):
        self.lr = lr
        self.parameters = parameters

    def update(self):
        for param in self.parameters:
            param.value = param.value - self.lr * param.grad


class MomentumSGD(object):
    def __init__(self, parameters, lr, rho = 0.85):
        self.lr= lr
        self.rho = rho
        self.parameters = parameters
        self.r=[]
        for param in parameters:
            self.r.append(torch.zeros_like(param.value))
    def update(self):
        temp = []
        for param,r in zip(self.parameters,self.r):
            r = self.rho * r - self.lr * param.grad
            temp.append(r)
            param.value =param.value + r
        self.r = temp

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


class AdaGrad(object):
    def __init__(self, parameters, lr,delta = 0.1):
        self.lr = lr
        self.delta=delta
        self.parameters = parameters
        self.r=[]
        for param in parameters:
            if param != []:
                self.r.append(torch.zeros_like(param[0]))
            else:
                self.r.append([])

    def update(self, parameters):
        temp = []
        for n in range(len(self.r)//2):
            param = parameters[n*2:2*n+2]
            new_p = []
            for i, p in enumerate(param):
                if p != []:
                    self.r[n*2+i] = self.r[n*2+i]+ torch.mul(p[1], p[1])
                    p[0] =p[0]-self.lr * p[1] /(self.delta + torch.sqrt(self.r[n*2+i]))
                new_p.append(p)
            temp.append(new_p)
        return temp
