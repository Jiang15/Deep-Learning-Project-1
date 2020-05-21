from torch import  nn
from Project1.SiameseNet.model import Siamese
from Project1.helpers import get_train_stats
from dlc_practical_prologue import generate_pair_sets
from torch.utils.data import DataLoader

N = 1000
train_input, train_target, train_class, test_input, test_target, test_class = generate_pair_sets(N)
# Data loaders
batch_size = 100
train_loader = DataLoader(list(zip(train_input, train_target, train_class)), batch_size)
test_loader = DataLoader(list(zip(test_input, test_target, test_class)), batch_size)
nb_channels = 2 # input channel
nb_digits = 10 # number of digit classes
nb_class = 2 # number of output classes

cross_entropy = nn.CrossEntropyLoss()
# reg = 0.001
# lr = 0.0004# Add learning rate
# AL_weight = 1
reg = [0.001, 0.001, 0.001]
lr = [0.01, 0.01, 0.01]# Add learning rate
AL_weight = 1
weight_sharing = [False, True, False]
auxiliary_loss = [False, False, True]
gamma = [0, 0, 0]
model = Siamese
epochs = 25
trial =  11
mean_tr = []
mean_te = []
std_tr = []
std_te = []
for i in range(len(auxiliary_loss)):
    mean_acc_tr, std_acc_tr, mean_acc_te, std_acc_te = get_train_stats(model, lr[i], reg[i], cross_entropy, AL_weight = AL_weight, trial = trial, epochs = epochs,  gamma = gamma[i], weight_sharing = weight_sharing[i], auxiliary_loss = auxiliary_loss[i])
    mean_tr.append(mean_acc_tr)
    mean_te.append(mean_acc_te)
    std_tr.append(std_acc_tr)
    std_te.append(std_acc_te)

for j in range(len(auxiliary_loss)):
    print("Auxiliaru loss: ", auxiliary_loss[j], ", weight sharing", weight_sharing[j],
          ", Train Accuracy: Mean = %.2f" % mean_tr[j], ", STD = %.2f" % std_tr[j], ", Test Accuracy: Mean = %.2f" % mean_te[j], "STD = %.2f" % std_te[j])
