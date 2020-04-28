import math

import torch

from Project2.dataset import generate_disc_set
from Project2.layers import Linear,  Relu
from Project2.loss_func import MSELoss
from Project2.optimizers import  SGD1
from Project2.Sequential import Sequential
from matplotlib import pyplot as plt
import numpy as np
def  generate_disc_set(nb):
    pts= torch.empty(nb, 2).uniform_()
    label=np.sqrt((pts[:,0]-0.5).pow(2) + (pts[:,1]-0.5).pow(2))> (1 / np.sqrt(2 * math.pi))
    target = torch.zeros(nb, 2)
    for i in range(len(label)):
        if label[i] == 0:
            target[i][0] = 1
        else:
            target[i][1] = 1

    return pts, target

train_input, train_target= generate_disc_set (1000)
test_input, test_target = generate_disc_set (1000)
mean, std = train_input.mean(), train_input.std()
train_input.sub_(mean).div_(std)
test_input.sub_(mean).div_(std)

train_input, train_target= generate_disc_set (1000)
test_input, test_target = generate_disc_set (1000)
mean, std = train_input.mean(), train_input.std()
train_input.sub_(mean).div_(std)
test_input.sub_(mean).div_(std)

def train(model, Loss, optimizer, input, target, nb_epochs,batch_size):
    loss_history = []
    for e in range(nb_epochs):
        loss_e = 0.
        for b in range(0, input.shape[0], batch_size):
            output = model.forward(input.narrow(0, b, batch_size))
            loss = Loss.forward(output, target.narrow(0, b, batch_size))
            loss_e += loss.item()
            #loss_history.append(loss_e)
            model.zero_grad()
            tmp = Loss.backward(output, target.narrow(0, b, batch_size))
            model.backward(tmp)
            optimizer.step()
        loss_history.append(loss_e)
    return loss_history

model = Sequential(Linear(2,25),Relu(),Linear(25,50),Relu(),Linear(50,25), Relu(),Linear(25,2))
Loss = MSELoss()
optimizer = SGD1(parameters = model.param(), lr = 0.1)
loss_train=train(model,Loss,optimizer,train_input,train_target, nb_epochs = 50, batch_size=100)
plt.plot(loss_train)
plt.show()

