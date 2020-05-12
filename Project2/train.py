import math

import torch

from Project2.dataset import generate_disc_set
from Project2.layers import Linear, Relu, Tanh, Leaky_Relu, Elu
from Project2.loss_func import MSELoss
from Project2.optimizers import  SGD
from Project2.Sequential import Sequential
from matplotlib import pyplot as plt
from Project2.helpers import normalize, plotLossAcc, calc_accuracy
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
train_input = normalize(train_input)
test_input = normalize(test_input)


def train(model, Loss, optimizer, input_tr, target_tr, input_te, target_te, nb_epochs,batch_size):
    loss_history_tr = []
    loss_history_te = []
    acc_history_tr = []
    acc_history_te = []
    for e in range(nb_epochs):
        loss_e_tr = 0.
        loss_e_te = 0.
        for b in range(0, input_tr.shape[0], batch_size):
            output = model.forward(input_tr.narrow(0, b, batch_size))
#             loss = Loss.forward(output, target_tr.narrow(0, b, batch_size))
#             loss_e_tr += loss.item()
            model.zero_grad()
            tmp = Loss.backward(output, target_tr.narrow(0, b, batch_size))
            model.backward(tmp)
            optimizer.update()

        output_tr = model.forward(input_tr)
        loss_e_tr = Loss.forward(output_tr, target_tr).item()
        output_te = model.forward(input_te)
        loss_e_te = Loss.forward(output_te, target_te).item()
        loss_history_tr.append(loss_e_tr)
        loss_history_te.append(loss_e_te)
        acc_history_tr.append(calc_accuracy(model, input_tr, target_tr))
        acc_history_te.append(calc_accuracy(model, input_te, target_te))

    return loss_history_tr, loss_history_te, acc_history_tr, acc_history_te


def cross_validation(k_fold, lr_set, input, target):

    interval = int(train_input.shape[0]/ k_fold)
    indices = torch.randperm(input.shape[0])
    nb_epochs = 50
    batch_size = 50
    loss = MSELoss()
    loss_tr_set = []
    loss_te_set = []
    acc_tr_set = []
    acc_te_set = []
#     min_loss_te = float('inf')
    max_acc_te = 0.
    best_lr = 0
    for lr in lr_set:
        loss_tr = 0
        loss_te = 0
        acc_tr = 0
        acc_te = 0
        for k in range(k_fold):
            model = Sequential(Linear(2,25),Elu(),Linear(25,50),Leaky_Relu(),Linear(50,25), Elu(),Linear(25,2))
            optimizer = SGD(parameters = model.param(), lr = lr.item())
            train_indices = indices[k*interval:(k+1)*interval]
            input_te = input[train_indices]
            target_te = target[train_indices]
            residual = torch.cat((indices[0:k*interval],indices[(k+1)*interval:]),0)
            input_tr = input[residual]
            target_tr = target[residual]
            loss_tr_temp, loss_te_temp, acc_tr_temp, acc_te_temp=train(model,loss,optimizer,input_tr, target_tr, input_te, target_te, nb_epochs = nb_epochs, batch_size=batch_size)
            loss_tr += loss_tr_temp[-1]
            loss_te += loss_te_temp[-1]
            acc_tr += acc_tr_temp[-1]
            acc_te += acc_te_temp[-1]

        loss_tr_set.append(loss_tr/k_fold)
        loss_te_set.append(loss_te/k_fold)
        acc_tr_set.append(acc_tr/k_fold)
        acc_te_set.append(acc_te/k_fold)

        if acc_te_set[-1] > max_acc_te:
            max_acc_te = acc_te_set[-1]
            best_lr = lr
    return best_lr.item(), loss_tr_set, loss_te_set, acc_tr_set, acc_te_set

# lr cross validation (find a way to pass model in the function and allow it to reinitialize in the for loop?)
lr_set = torch.logspace(-2, 0.01, 20)
k_fold = 10
# plt.show()

# Train with best lr found
model = Sequential(Linear(2,25),Elu(),Linear(25,50),Leaky_Relu(),Linear(50,25), Elu(),Linear(25,2))
loss = MSELoss()
best_lr, loss_tr_set, loss_te_set, acc_tr_set, acc_te_set = cross_validation(k_fold,lr_set, train_input,train_target)
print(best_lr)
optimizer = SGD(parameters = model.param(), lr = best_lr)
plotLossAcc(loss_tr_set, loss_te_set, acc_tr_set, acc_te_set, lr_set, "Learning Rate")

nb_epochs = 50
batch_size = 50
loss_train,loss_test, acc_train, acc_test =train(model,loss,optimizer,train_input,train_target,test_input,test_target, nb_epochs = nb_epochs, batch_size=batch_size)
epochs = torch.arange(nb_epochs)
plotLossAcc(loss_train, loss_test, acc_train, acc_test, epochs, "Epochs")
plt.show()

model = Sequential(Linear(2,25),Tanh(),Linear(25,50),Tanh(),Linear(50,25), Tanh(),Linear(25,2))
loss = MSELoss()
best_lr, loss_tr_set, loss_te_set, acc_tr_set, acc_te_set = cross_validation(k_fold,lr_set, train_input,train_target)
print(best_lr)
optimizer = SGD(parameters = model.param(), lr = best_lr)
plotLossAcc(loss_tr_set, loss_te_set, acc_tr_set, acc_te_set, lr_set, "Learning Rate")
nb_epochs = 50
batch_size = 50

loss_train,loss_test, acc_train, acc_test =train(model,loss,optimizer,train_input,train_target,test_input,test_target, nb_epochs = nb_epochs, batch_size=batch_size)
epochs = torch.arange(nb_epochs)
plotLossAcc(loss_train, loss_test, acc_train, acc_test, epochs, "Epochs")
plt.show()
