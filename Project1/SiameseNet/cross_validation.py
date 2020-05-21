from torch import nn
from Project1.RCNN.model import CNN
from Project1.SiameseNet.model import Siamese
from Project1.helpers import cross_validation
from matplotlib import pyplot as plt

weight_sharing = True #[False, True, False, True]
auxiliary_loss = False #[False, False, True, True]
criterion = nn.CrossEntropyLoss()

reg_set = [0.001]
lr_set = [0.008, 0.01, 0.012]
gamma_set = [0.1]
epochs = 25
k_fold = 10

AL_weight = 1
model = Siamese

best_lr, best_reg, best_gamma, loss_tr_set, loss_te_set, accuracy_tr_set, accuracy_te_set, train_info_mean_set = cross_validation(k_fold, lr_set, reg_set, gamma_set, model, criterion, AL_weight, epochs, batch_size = 100, test_every = 9, weight_sharing = weight_sharing, auxiliary_loss = auxiliary_loss)
print('best lr = ', best_lr)
print('best reg = ', best_reg)
print('best gamma = ', best_gamma)

plt.show()
