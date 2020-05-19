import torch

# from Project2.dataset import generate_disc_set
from Project2.layers import Linear, Relu, Tanh, Leaky_Relu, Elu, Sigmoid
from Project2.loss_func import MSELoss, BCELoss
from Project2.optimizers2 import  SGD, Adam, MomentumSGD, AdaGrad
from Project2.Sequential import Sequential
from matplotlib import pyplot as plt
from Project2.helpers import normalize, plotLossAcc, generate_disc_set, train, cross_validation


########################################################################################################################
# initial setups


nb_epochs = 50
batch_size = 50

# Generate training and test data sets and normalize
train_input, train_target= generate_disc_set(1000)
test_input, test_target = generate_disc_set(1000)
train_input = normalize(train_input)
test_input = normalize(test_input)

# K-fold cross validation to optimize learning rate over range lr_set
lr_set = torch.logspace(-3, -0.1, 5)
k_fold = 5

# create models
model_LReLu = Sequential(Linear(2,25),Elu(),Linear(25,50),Leaky_Relu(),Linear(50,25), Elu(),Linear(25,2))
model_Tanh = Sequential(Linear(2,25),Tanh(),Linear(25,50),Tanh(),Linear(50,25), Tanh(),Linear(25,2), Sigmoid())
# model_Tanh = Sequential(Linear(2,25), Tanh(),Linear(25,2))

# set optimizer and loss
optimizer_name = AdaGrad
loss = MSELoss()

########################################################################################################################
# cross validation and plot
# best_lr_LReLu, loss_tr_set_LReLu, loss_te_set_LReLu, acc_tr_set_LReLu, acc_te_set_LReLu = cross_validation(model_LReLu, optimizer_name, nb_epochs, batch_size, loss, k_fold,lr_set, train_input,train_target)
# print(best_lr_LReLu)
# plotLossAcc(loss_tr_set_LReLu, loss_te_set_LReLu, acc_tr_set_LReLu, acc_te_set_LReLu, lr_set, "Learning Rate")

best_lr_Tanh, loss_tr_set_Tanh, loss_te_set_Tanh, acc_tr_set_Tanh, acc_te_set_Tanh = cross_validation(model_Tanh, optimizer_name, nb_epochs, batch_size, loss, k_fold,lr_set, train_input,train_target)
print(best_lr_Tanh)
plotLossAcc(loss_tr_set_Tanh, loss_te_set_Tanh, acc_tr_set_Tanh, acc_te_set_Tanh, lr_set, "Learning Rate")
########################################################################################################################
# train models with best learning rates found

# reinitialize models
# model_LReLu.reset()
model_Tanh.reset()

# set up optimizers
# optimizer_LReLu = optimizer_name(parameters = model_LReLu.param(), lr = best_lr_LReLu)
optimizer_Tanh = optimizer_name(parameters = model_Tanh.param(), lr = best_lr_Tanh)


# model training
# loss_train_LReLu, loss_test_LReLu, acc_train_LReLu, acc_test_LReLu = train(model_LReLu, loss, optimizer_LReLu,train_input,train_target,test_input,test_target, nb_epochs = nb_epochs, batch_size=batch_size)
loss_train_Tanh, loss_test_Tanh, acc_train_Tanh, acc_test_Tanh = train(model_Tanh, loss, optimizer_Tanh,train_input,train_target,test_input,test_target, nb_epochs = nb_epochs, batch_size=batch_size)

# plots
epochs = torch.arange(nb_epochs)
# plotLossAcc(loss_train_LReLu, loss_test_LReLu, acc_train_LReLu, acc_test_LReLu, epochs, "Epochs")
plotLossAcc(loss_train_Tanh, loss_test_Tanh, acc_train_Tanh, acc_test_Tanh, epochs, "Epochs")

plt.show()
