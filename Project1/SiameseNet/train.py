from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from Project1.SiameseNet.model import Siamese
from Project1.helpers import train, plot_train_info
from dlc_practical_prologue import generate_pair_sets



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


reg = 0.001
lr = 0.001# Add learning rate decay
epochs = 23
weight_sharing_CNN = False
weight_sharing_FC = False
auxiliary_loss = False
AL_weight = 1
weight_sharing_CNN = False
weight_sharing_FC = False
auxiliary_loss = True


net = Siamese(nb_channels, nb_class, weight_sharing_CNN, weight_sharing_FC, auxiliary_loss)

train_info_AL = train(train_loader, test_loader,
                         model = net,
                         optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=reg),
                         criterion = cross_entropy, AL_weight = AL_weight,
                         epochs = epochs, test_every=5, weight_sharing=False, auxiliary_loss = auxiliary_loss)




net = Siamese(nb_channels, nb_class, weight_sharing_CNN, weight_sharing_FC, auxiliary_loss)

train_info = train(train_loader, test_loader,
                     model = net,
                     optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay = reg),
                     criterion = cross_entropy, AL_weight = AL_weight,
                     epochs = epochs, test_every = 5, weight_sharing= False, auxiliary_loss = auxiliary_loss)


weight_sharing_CNN = True
weight_sharing_FC = True
auxiliary_loss = False

net = Siamese(nb_channels, nb_class, weight_sharing_CNN, weight_sharing_FC, auxiliary_loss)

train_info_WS = train(train_loader, test_loader,
                 model = net,
                 optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=reg),
                 criterion = cross_entropy, AL_weight = AL_weight,
                 epochs = epochs, test_every=5, weight_sharing= True,  auxiliary_loss = auxiliary_loss)

plot_train_info(train_info, False)
plot_train_info(train_info_WS, False)
plot_train_info(train_info_AL, True)
