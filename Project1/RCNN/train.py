from torch import nn
from Project1.RCNN.model import RCNN
from Project1.helpers import plot_train_info, get_train_stats
from torchsummary import summary
import matplotlib.pyplot as plt



# Data loaders
batch_size = 1
nb_channels = 2 # input channel
nb_digits = 10 # number of digit classes
nb_class = 2 # number of output classes

cross_entropy = nn.CrossEntropyLoss()

RNN_model = RCNN(nb_channels, nb_class,True, True, K = 32)
summary(RNN_model, input_size=(2, 14, 14))
RNN_model = RCNN(nb_channels, nb_class,False, False, K = 32)
summary(RNN_model, input_size=(2, 14, 14))

reg = 0.01 #0.025 0.02 0.04 0.015
lr = 0.005# 0.005 0.001 0.003 Add learning rate decay !
epochs = 25

mean_tr = []
mean_te = []
std_tr = []
std_te = []



weight_sharing_recurr =[False]#[False, True, False, True]
auxiliary_loss = [True]#[False, False, True, True]
AL_weight = 0.01
model = RCNN

for i in range(len(auxiliary_loss)):
    mean_acc_tr, std_acc_tr, mean_acc_te, std_acc_te, train_info_mean = get_train_stats(model, lr, reg, cross_entropy, AL_weight = AL_weight, epochs = epochs, gamma = 0, weight_sharing = weight_sharing_recurr[i], auxiliary_loss = auxiliary_loss[i])
    plot_train_info(train_info_mean, weight_sharing_recurr[i], auxiliary_loss[i])
    mean_tr.append(mean_acc_tr)
    mean_te.append(mean_acc_te)
    std_tr.append(std_acc_tr)
    std_te.append(std_acc_te)

for j in range(len(auxiliary_loss)):
    print("AL: ", auxiliary_loss[j], "WS: ", weight_sharing_recurr[j], \
          "Train Accuracy: Mean = ", mean_tr[j], "STD =", std_tr[j], "Test Accuracy: Mean = ", mean_te[j], "STD =", std_te[j])

plt.show()

