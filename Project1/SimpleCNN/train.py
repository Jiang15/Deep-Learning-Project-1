from torch import nn
from torch import optim

from torch.utils.data import DataLoader

from Project1.SimpleCNN.model import CNN
from Project1.helpers import train, plot_train_info
from dlc_practical_prologue import generate_pair_sets

from torchsummary import summary

N = 1000
train_input, train_target, train_class, test_input, test_target, test_class = generate_pair_sets(N)
# Data loaders
batch_size = 100
train_loader = DataLoader(list(zip(train_input, train_target, train_class)), batch_size)
test_loader = DataLoader(list(zip(test_input, test_target, test_class)), batch_size)
nb_channels = 2  # input channel
nb_digits = 10  # number of digit classes
nb_class = 2  # number of output classes

cross_entropy = nn.CrossEntropyLoss()

#     def predict(self, output_class):
#         _, predicted_class = torch.max(output_class, 1)
#         return predicted_class
CNN_model = CNN(nb_channels, nb_class, auxiliary_loss=True)
summary(CNN_model, input_size=(2, 14, 14))
reg = 0.3
lr = 0.001  # Add learning rate decay
epochs = 25
auxiliary_loss = False
AL_weight = 0.3  # aux loss weight

net = CNN(nb_channels, nb_class, auxiliary_loss)
train_info = train(train_loader, test_loader,
                   model=net,
                   optimizer=optim.Adam(net.parameters(), lr=lr, weight_decay=reg),
                   criterion=cross_entropy, AL_weight=AL_weight,
                   epochs=epochs, test_every=5, auxiliary_loss=auxiliary_loss)

# reg = 0.5
# lr = 0.0005# Add learning rate decay
# epochs = 25
AL_weight = 0.5

auxiliary_loss = True
net = CNN(nb_channels, nb_class, auxiliary_loss)
train_info_AL = train(train_loader, test_loader,
                      model=net,
                      optimizer=optim.Adam(net.parameters(), lr=lr, weight_decay=reg),
                      criterion=cross_entropy, AL_weight=AL_weight,
                      epochs=epochs, test_every=5, auxiliary_loss=auxiliary_loss)

plot_train_info(train_info, False)
plot_train_info(train_info_AL, True)
