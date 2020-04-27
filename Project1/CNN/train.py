from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from Project1.CNN.model import CNN
from Project1.helpers import train, plot_train_info, get_train_stats
from dlc_practical_prologue import generate_pair_sets
import matplotlib.pyplot as plt

from torchsummary import summary

# N = 1000
# train_input, train_target, train_class, test_input, test_target, test_class = generate_pair_sets(N)
# # Data loaders
# batch_size = 100
# train_loader = DataLoader(list(zip(train_input, train_target, train_class)), batch_size, shuffle = True)
# test_loader = DataLoader(list(zip(test_input, test_target, test_class)), batch_size, shuffle = True)
nb_channels = 2  # input channel
nb_digits = 10  # number of digit classes
nb_class = 2  # number of output classes
cross_entropy = nn.CrossEntropyLoss()
CNN_model = CNN(nb_channels, nb_class, weight_sharing=True, auxiliary_loss=True)
summary(CNN_model, input_size=(2, 14, 14))

reg = [0,0]
lr = [0.005,0.005]  # 0.001 Add learning rate decay
epochs = 25
AL_weight = 0.1  # 0.3 aux loss weight
gamma = [0,0]
model = CNN

auxiliary_loss = [False,True]
model = CNN

mean_tr = []
mean_te = []
std_tr = []
std_te = []



for i in range(len(auxiliary_loss)):
    mean_acc_tr, std_acc_tr, mean_acc_te, std_acc_te, train_info_mean = get_train_stats(model, lr[i], reg[i], cross_entropy, AL_weight = AL_weight, epochs = epochs,  gamma = gamma[i], weight_sharing = False, auxiliary_loss = auxiliary_loss[i])
    plot_train_info(train_info_mean, weight_sharing = False, auxiliary_loss = auxiliary_loss[i])
    mean_tr.append(mean_acc_tr)
    mean_te.append(mean_acc_te)
    std_tr.append(std_acc_tr)
    std_te.append(std_acc_te)

for j in range(len(auxiliary_loss)):
    print("AL: ", auxiliary_loss[j],\
          "Train Accuracy: Mean = ", mean_tr[j], "STD =", std_tr[j], "Test Accuracy: Mean = ", mean_te[j], "STD =", std_te[j])

plt.show()
