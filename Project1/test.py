from torch import nn
from Project1.models.Siamese import Siamese
from Project1.models.RCNN import CNN
from Project1.models.FNN import FNN
from Project1.models.Resnet import ResNet

from Project1.helpers import get_train_stats, cross_validation
from dlc_practical_prologue import generate_pair_sets
from torch.utils.data import DataLoader

########################################################################################################################
# Initial setups

run_cross_validation = False # boolean flag: True to run cross validation; False to run training and testing
# generate train and test sets
N = 1000
train_input, train_target, train_class, test_input, test_target, test_class = generate_pair_sets(N)

# Data loaders
batch_size = 100
train_loader = DataLoader(list(zip(train_input, train_target, train_class)), batch_size)
test_loader = DataLoader(list(zip(test_input, test_target, test_class)), batch_size)
nb_channels = 2 # input channel
nb_digits = 10 # number of digit classes
nb_class = 2 # number of output classes

# loss function
cross_entropy = nn.CrossEntropyLoss()

epochs = 25 # number of epochs

# cases with/without weight sharing and auxiliary loss
weight_sharing = [False, True, False]
auxiliary_loss = [False, False, True]
# auxiliary loss weighting
AL_weight = 1

mean_tr = []
mean_te = []
std_tr = []
std_te = []

########################################################################################################################
# FNN: Run cross validation or training and testing
model = FNN
print("FNN Model")

if run_cross_validation: # Run cross validation to help select optimal hyperparameter
    k_fold = 5
    lr_set = [0.0001, 0.001, 0.01, 0.1] # learning rate range for cross validation
    reg_set = [0, 0.1, 0.2, 0.3] # weight decay factor range
    gamma_set = [0, 0.1] # learning rate scheduler multiplicative factor range
    for i in range(len(auxiliary_loss)):
        cross_validation(k_fold, lr_set, reg_set, gamma_set, model, cross_entropy, AL_weight, epochs,
                         batch_size = batch_size,  weight_sharing = weight_sharing[i], auxiliary_loss = auxiliary_loss[i])

else: # train and test the model
    trial =  11 # number of trials
    # hyperparameters for training and testing
    reg = [0.001, 0.001, 0.001] # weight decay factor
    lr = [0.01, 0.01, 0.01]# learning rate
    gamma = [0, 0, 0] # learing rate scheduler's multiplicative factor

    for i in range(len(auxiliary_loss)):
        mean_acc_tr, std_acc_tr, mean_acc_te, std_acc_te = get_train_stats(model, lr[i], reg[i], cross_entropy, AL_weight = AL_weight, trial = trial, epochs = epochs,  gamma = gamma[i], weight_sharing = weight_sharing[i], auxiliary_loss = auxiliary_loss[i])
        mean_tr.append(mean_acc_tr)
        mean_te.append(mean_acc_te)
        std_tr.append(std_acc_tr)
        std_te.append(std_acc_te)

if not run_cross_validation: # if not cross validation, print the test results
    for j in range(len(auxiliary_loss)):
        print("Auxiliary loss: ", auxiliary_loss[j], ", weight sharing", weight_sharing[j],
              ", Train Accuracy: Mean = %.2f" % mean_tr[j], ", STD = %.2f" % std_tr[j], ", Test Accuracy: Mean = %.2f" % mean_te[j], "STD = %.2f" % std_te[j])

########################################################################################################################
# Siamese: Run cross validation or training and testing
model = Siamese
print("SiameseNet Model")
if run_cross_validation: # Run cross validation to help select optimal hyperparameter
    k_fold = 5
    lr_set = [0.0001, 0.001, 0.01, 0.1] # learning rate range for cross validation
    reg_set = [0, 0.1, 0.2, 0.3] # weight decay factor range
    gamma_set = [0, 0.1] # learning rate scheduler multiplicative factor range
    for i in range(len(auxiliary_loss)):
        cross_validation(k_fold, lr_set, reg_set, gamma_set, model, cross_entropy, AL_weight, epochs,
                         batch_size = batch_size,  weight_sharing = weight_sharing[i], auxiliary_loss = auxiliary_loss[i])

else: # train and test the model
    trial =  11 # number of trials
    # hyperparameters for training and testing
    reg = [0.001, 0.001, 0.001] # weight decay factor
    lr = [0.01, 0.01, 0.01]# learning rate
    gamma = [0, 0, 0] # learing rate scheduler's multiplicative factor

    for i in range(len(auxiliary_loss)):
        mean_acc_tr, std_acc_tr, mean_acc_te, std_acc_te = get_train_stats(model, lr[i], reg[i], cross_entropy, AL_weight = AL_weight, trial = trial, epochs = epochs,  gamma = gamma[i], weight_sharing = weight_sharing[i], auxiliary_loss = auxiliary_loss[i])
        mean_tr.append(mean_acc_tr)
        mean_te.append(mean_acc_te)
        std_tr.append(std_acc_tr)
        std_te.append(std_acc_te)

if not run_cross_validation: # if not cross validation, print the test results
    for j in range(len(auxiliary_loss)):
        print("Auxiliary loss: ", auxiliary_loss[j], ", weight sharing", weight_sharing[j],
              ", Train Accuracy: Mean = %.2f" % mean_tr[j], ", STD = %.2f" % std_tr[j], ", Test Accuracy: Mean = %.2f" % mean_te[j], "STD = %.2f" % std_te[j])

########################################################################################################################
# CNN: Run cross validation or training and testing
print("CNN Model")
model = CNN

if run_cross_validation: # Run cross validation to help select optimal hyperparameter
    k_fold = 5
    lr_set = [0.0001, 0.001, 0.01, 0.1] # learning rate range for cross validation
    reg_set = [0, 0.1, 0.2, 0.3] # weight decay factor range
    gamma_set = [0, 0.1] # learning rate scheduler multiplicative factor range
    for i in range(len(auxiliary_loss)):
        cross_validation(k_fold, lr_set, reg_set, gamma_set, model, cross_entropy, AL_weight, epochs,
                         batch_size = batch_size,  weight_sharing = weight_sharing[i], auxiliary_loss = auxiliary_loss[i])

else: # train and test the model
    trial =  11 # number of trials
    # hyperparameters for training and testing
    reg = [0.15, 0.1, 0.3] # weight decay factor
    lr = [0.0015, 0.0015, 0.0025]# learning rate
    gamma = [0.2, 0.1, 0.1] # learing rate scheduler's multiplicative factor


    for i in range(len(auxiliary_loss)):
        mean_acc_tr, std_acc_tr, mean_acc_te, std_acc_te = get_train_stats(model, lr[i], reg[i], cross_entropy, AL_weight = AL_weight, trial = trial, epochs = epochs,  gamma = gamma[i], weight_sharing = weight_sharing[i], auxiliary_loss = auxiliary_loss[i])
        mean_tr.append(mean_acc_tr)
        mean_te.append(mean_acc_te)
        std_tr.append(std_acc_tr)
        std_te.append(std_acc_te)

if not run_cross_validation: # if not cross validation, print the test results
    for j in range(len(auxiliary_loss)):
        print("Auxiliary loss: ", auxiliary_loss[j], ", weight sharing", weight_sharing[j],
              ", Train Accuracy: Mean = %.2f" % mean_tr[j], ", STD = %.2f" % std_tr[j], ", Test Accuracy: Mean = %.2f" % mean_te[j], "STD = %.2f" % std_te[j])

########################################################################################################################
# ResNet: Run cross validation or training and testing
print("ResNet Model")
model = ResNet

if run_cross_validation: # Run cross validation to help select optimal hyperparameter
    k_fold = 5
    lr_set = [0.0001, 0.001, 0.01, 0.1] # learning rate range for cross validation
    reg_set = [0, 0.1, 0.2, 0.3] # weight decay factor range
    gamma_set = [0, 0.1] # learning rate scheduler multiplicative factor range
    for i in range(len(auxiliary_loss)):
        cross_validation(k_fold, lr_set, reg_set, gamma_set, model, cross_entropy, AL_weight, epochs,
                         batch_size = batch_size,  weight_sharing = weight_sharing[i], auxiliary_loss = auxiliary_loss[i])

else: # train and test the model
    trial =  11 # number of trials
    # hyperparameters for training and testing
    reg = [0.001, 0.001, 0.001] # weight decay factor
    lr = [0.01, 0.01, 0.01]# learning rate
    gamma = [0, 0, 0] # learing rate scheduler's multiplicative factor

    for i in range(len(auxiliary_loss)):
        mean_acc_tr, std_acc_tr, mean_acc_te, std_acc_te = get_train_stats(model, lr[i], reg[i], cross_entropy, AL_weight = AL_weight, trial = trial, epochs = epochs,  gamma = gamma[i], weight_sharing = weight_sharing[i], auxiliary_loss = auxiliary_loss[i])
        mean_tr.append(mean_acc_tr)
        mean_te.append(mean_acc_te)
        std_tr.append(std_acc_tr)
        std_te.append(std_acc_te)

if not run_cross_validation: # if not cross validation, print the test results
    for j in range(len(auxiliary_loss)):
        print("Auxiliary loss: ", auxiliary_loss[j], ", weight sharing", weight_sharing[j],
              ", Train Accuracy: Mean = %.2f" % mean_tr[j], ", STD = %.2f" % std_tr[j], ", Test Accuracy: Mean = %.2f" % mean_te[j], "STD = %.2f" % std_te[j])
