from torch import nn
from Project1.RCNN.model import CNN
from Project1.helpers import plot_train_info, get_train_stats
from torchsummary import summary
import matplotlib.pyplot as plt


# Data loaders
batch_size = 100
nb_digits = 10 # number of digit classes
nb_class = 2 # number of output classes

cross_entropy = nn.CrossEntropyLoss()

mean_tr = []
mean_te = []
std_tr = []
std_te = []



weight_sharing = [False] #[False, True, False, True]
auxiliary_loss = [False]#[False, False, True, True]


reg = [0.15] # TT: 0.3 0.002 0.1 al 0.6 (0.9272, std 0.0102) FT 0.3 0.002 0.1 1(0.9445, std 0.0098) # TF 0.1 0.0015 0.1 (0.8766, std 0.0127)
lr = [0.001] # FF 0.15 0.0015 0.1 (0.8618, std 0.0129)
gamma = [0.1]
epochs = 25
trial =  11
AL_weight = 0.6
model = CNN



for i in range(len(auxiliary_loss)):
    mean_acc_tr, std_acc_tr, mean_acc_te, std_acc_te, train_info_mean = get_train_stats(model, lr[i], reg[i], cross_entropy, AL_weight = AL_weight, epochs = epochs, trial = trial, test_every=10, gamma = gamma[i], weight_sharing = weight_sharing[i], auxiliary_loss = auxiliary_loss[i])
    plot_train_info(train_info_mean, weight_sharing[i], auxiliary_loss[i])
    mean_tr.append(mean_acc_tr)
    mean_te.append(mean_acc_te)
    std_tr.append(std_acc_tr)
    std_te.append(std_acc_te)

for j in range(len(auxiliary_loss)):
    print("AL: ", auxiliary_loss[j], "WS: ", weight_sharing[j], \
          "Train Accuracy: Mean = ", mean_tr[j], "STD =", std_tr[j], "Test Accuracy: Mean = ", mean_te[j], "STD =", std_te[j])

plt.show()

