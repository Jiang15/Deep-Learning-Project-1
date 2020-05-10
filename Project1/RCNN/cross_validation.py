from torch import nn
from Project1.RCNN.model import CNN, CNN_recurr
from Project1.helpers import cross_validation, plotCV
from matplotlib import pyplot as plt




weight_sharing = False #[False, True, False, True]
auxiliary_loss = False#[False, False, True, True]
criterion = nn.CrossEntropyLoss()

reg_set = [0.025, 0.03, 0.035] # CNN: 0.0.863 TF: 0.02 0.0018 0.7 (0.854) # 0.839 FF: 0.03 0.0022 0.2 (0.836) # FT: 0.868 0.06 0.0015 0.3 al 0.3 (0.8605)  TT: 0.893 0.085 0.0015 0.3 al 0.4 (0.8924)
lr_set = [0.002, 0.0022, 0.0024]
gamma_set = [0.2]
epochs = 25
k_fold = 5

AL_weight = 0.4
model = CNN

best_lr, best_reg, best_gamma, loss_tr_set, loss_te_set, accuracy_tr_set, accuracy_te_set, train_info_mean_set = cross_validation(k_fold, lr_set, reg_set, gamma_set, model, criterion, AL_weight, epochs, batch_size = 100, test_every = 8, weight_sharing = weight_sharing, auxiliary_loss = auxiliary_loss)
print('best lr = ', best_lr)
print('best reg = ', best_reg)
print('best gamma = ', best_gamma)


plotCV(loss_tr_set, loss_te_set, accuracy_tr_set, accuracy_te_set, reg_set, lr_set, reg_set.index(best_reg), 'Learning Rate', 'Weight Decay = ')
plotCV(loss_tr_set, loss_te_set, accuracy_tr_set, accuracy_te_set, lr_set, reg_set, lr_set.index(best_lr), 'Weight Decay', 'Learning Rate = ')



plt.show()
