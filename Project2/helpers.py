import torch
from matplotlib import pyplot as plt


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

    ax1.legend(['train_loss','test_loss'])
    ax2.legend(['train_acc', 'test_acc'],loc='lower right')

    # plt.show()


def calc_accuracy(model, input, target):
    target = torch.argmax(target,1)
    total = len(target)
    output = model.forward(input)
    _, pred = torch.max(output, 1)
    correct = (pred == target).sum().item()
    return correct / total

#def train():





