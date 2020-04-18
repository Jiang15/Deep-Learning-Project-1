import torch
from torch import nn


class Siamese(nn.Module):
    def __init__(self, input_channels, output_channels, weight_sharing_CNN, weight_sharing_FC , auxiliary_loss):

        super().__init__()
        self.weight_sharing_CNN = weight_sharing_CNN
        self.weight_sharing_FC = weight_sharing_FC
        self.auxiliary_loss = auxiliary_loss
        #self.fc6 = nn.Linear(10*2, 2)

        self.CNNnet1 = nn.Sequential(nn.Conv2d(1, 32, 3),  nn.ReLU(), nn.MaxPool2d(2,2), nn.Dropout2d(0.3), nn.Conv2d(32, 64, 3),  nn.ReLU(), nn.MaxPool2d(2,2), nn.Dropout2d(0.3))
        self.CNNnet2 = nn.Sequential(nn.Conv2d(1, 32, 3), nn.ReLU(), nn.MaxPool2d(2,2), nn.Dropout2d(0.3), nn.Conv2d(32, 64, 3),  nn.ReLU(), nn.MaxPool2d(2,2), nn.Dropout2d(0.3))
        self.FCnet1 = nn.Sequential(nn.Linear(64 * 2 * 2, 50), nn.ReLU(), nn.Linear(50, 10))
        self.FCnet2 = nn.Sequential(nn.Linear(64 * 2 * 2, 50), nn.ReLU(), nn.Linear(50, 10))
        #self.combine = nn.Sequential(nn.Linear(64*2*2*2, 100), nn.ReLU(), nn.Linear(100, 25), nn.ReLU(), nn.Linear(25, 2))
        self.combine = nn.Sequential(nn.ReLU(), nn.Linear(10*2, 2))

    def forward(self, x):
        if self.weight_sharing_CNN == True:
            y1 = self.CNNnet1(torch.unsqueeze(x[:,0],dim=1))
            y2 = self.CNNnet1(torch.unsqueeze(x[:,1],dim=1))

        else:
            y1 = self.CNNnet1(torch.unsqueeze(x[:,0],dim=1))
            y2 = self.CNNnet2(torch.unsqueeze(x[:,1],dim=1))

        y1 = y1.view(-1, 64 *2 *2)
        y2 = y2.view(-1, 64 *2 *2)



        if self.weight_sharing_FC == True:
            y1 = self.FCnet1(y1)
            y2 = self.FCnet1(y2)

        else:
            y1 = self.FCnet1(y1)
            y2 = self.FCnet2(y2)
        y = torch.cat((y1, y2), dim = 1)
        #y = y.view(-1, 64*2*2*2)
        y = y.view(-1, 2*10)
        y = self.combine(y)

        if self.auxiliary_loss == True:
            return y1, y2, y
        else:
            return y
