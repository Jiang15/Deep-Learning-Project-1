from torch import nn
from Project1.RCNN.model import CNN, CNN_recurr
from Project1.helpers import plot_train_info, get_train_stats
from torchsummary import summary
import matplotlib.pyplot as plt



# Data loaders
batch_size = 100
nb_channels = 2 # input channel
nb_digits = 10 # number of digit classes
nb_class = 2 # number of output classes

cross_entropy = nn.CrossEntropyLoss()

RNN_model = CNN(nb_channels, nb_class,False, False, K = 32)
summary(RNN_model, input_size=(2, 14, 14))
RNN_model = CNN(nb_channels, nb_class,True, False, K = 32)
summary(RNN_model, input_size=(2, 14, 14))

# reg = [0.04,0.06,0,0]
# lr = [0.006,0.006,0.015,0.01]
# epochs = 25
# gamma = [0.02, 0.02, 0.001, 0.001]


mean_tr = []
mean_te = []
std_tr = []
std_te = []



weight_sharing = [True] #[False, True, False, True]
auxiliary_loss = [True]#[False, False, True, True]


reg = [0.085] # CNN: 0.0.863 TF: 0.02 0.0018 0.7 (0.854) # 0.855 FF: 0.06 0.0017 0.5 25 (0.8458) # FT: 0.868 0.06 0.0015 0.3 al 0.3 (0.8605)  TT: 0.893 0.085 0.0015 0.3 al 0.4 (0.8924)
lr = [0.0015]# CNN_recurr: 0.0.86 TF: 0.06 0.0017 0.5 (0.)
gamma = [0.3]
epochs = 25

AL_weight = 0.4
model = CNN



for i in range(len(auxiliary_loss)):
    mean_acc_tr, std_acc_tr, mean_acc_te, std_acc_te, train_info_mean = get_train_stats(model, lr[i], reg[i], cross_entropy, AL_weight = AL_weight, epochs = epochs,test_every=10, gamma = gamma[i], weight_sharing = weight_sharing[i], auxiliary_loss = auxiliary_loss[i])
    plot_train_info(train_info_mean, weight_sharing[i], auxiliary_loss[i])
    mean_tr.append(mean_acc_tr)
    mean_te.append(mean_acc_te)
    std_tr.append(std_acc_tr)
    std_te.append(std_acc_te)

for j in range(len(auxiliary_loss)):
    print("AL: ", auxiliary_loss[j], "WS: ", weight_sharing[j], \
          "Train Accuracy: Mean = ", mean_tr[j], "STD =", std_tr[j], "Test Accuracy: Mean = ", mean_te[j], "STD =", std_te[j])

plt.show()

