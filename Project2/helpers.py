import torch
from matplotlib import pyplot as plt
import math
from torch import empty
from Project2.loss_func import MSELoss

torch.manual_seed(0)


def generate_disc_set(nb):
    pts = empty(nb, 2).uniform_(0, 1)
    label = (-((pts-0.5).pow(2).sum(1)-(1 / (2 * math.pi))).sign()+1)/2 # take reversed sign of subtraction with 2/pi then add 1 and divide by 2 then reverse
    target = torch.zeros(nb, 2)
    for i in range(len(label)):
        if label[i] == 0:
            target[i][0] = 1
        else:
            target[i][1] = 1

    return pts, target


def normalize(x):
    mean_x=x.mean()
    std_x=x.std()
    x=(x-mean_x)/std_x
    return x


# plot loss and accuracy on same plot
def plotLossAcc(loss_tr, loss_te, acc_tr, acc_te, x, xlabel):
    fig, ax1 = plt.subplots()
    color_tr = 'tab:green'
    color_te = 'tab:blue'
    color_tra = 'tab:orange'
    color_tea = 'tab:purple'
    ax1.set_xlabel(xlabel)
    ax1.plot(x,loss_tr,color=color_tr)
    ax1.plot(x,loss_te, color=color_te)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.plot(x, acc_tr, color=color_tra)
    ax2.plot(x, acc_te, color=color_tea)

    ax1.legend(['train_loss','validation_loss'])
    ax2.legend(['train_accuracy', 'validation_accuracy'],loc='lower right')

    ax1.set_title("Validation Accuracy = "+str(acc_te[-1]))
    # plt.show()


def calc_accuracy(model, input, target):
    target = torch.argmax(target,1)
    total = len(target)
    output = model.forward(input)
    _, pred = torch.max(output, 1)
    correct = (pred == target).sum().item()
    return correct / total


def train(model, Loss, optimizer, input_tr, target_tr, input_te, target_te, nb_epochs,batch_size):
    loss_history_tr = []
    loss_history_te = []
    acc_history_tr = []
    acc_history_te = []
    for e in range(nb_epochs):
        for b in range(0, input_tr.shape[0], batch_size):
            output = model.forward(input_tr.narrow(0, b, batch_size))
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


def cross_validation(model, optimizer_name, nb_epochs, batch_size, loss, k_fold, lr_set, input, target):
    interval = int(input.shape[0]/ k_fold)
    indices = torch.randperm(input.shape[0])

    loss_tr_set = []
    loss_te_set = []
    acc_tr_set = []
    acc_te_set = []
    max_acc_te = 0.
    best_lr = 0
    for i, lr in enumerate(lr_set):
        loss_tr = 0
        loss_te = 0
        acc_tr = 0
        acc_te = 0
        print("Running cross validation. Progress:  ", i/len(lr_set)*100, '%')

        for k in range(k_fold):
            model.reset()
            optimizer = optimizer_name(model = model, lr = lr.item())
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






