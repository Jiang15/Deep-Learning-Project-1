import torch
from torch import nn


class Siamese(nn.Module):
    def __init__(self, input_channels, output_channels, weight_sharing_CNN, weight_sharing_FC , auxiliary_loss ):

        super().__init__()
        self.weight_sharing_CNN = weight_sharing_CNN
        self.weight_sharing_FC = weight_sharing_FC
        self.auxiliary_loss = auxiliary_loss
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.dropout = nn.Dropout2d(0.3)
        self.fc1 = nn.Linear(64 * 2 * 2, 50)
        self.fc2 = nn.Linear(50, 10) #output
        self.fc3 = nn.Linear(64*2*2*2, 100 )
        self.fc4 = nn.Linear(100, 25)
        self.fc5 = nn.Linear(25, 2)
        self.fc6 = nn.Linear(10*2, 2)
        self.CNNnet1 = nn.Sequential(self.conv1, nn.ReLU(), self.pool, self.dropout, self.conv2, nn.ReLU(), self.pool, self.dropout)
        self.CNNnet2 = nn.Sequential(self.conv1, nn.ReLU(), self.pool, self.dropout, self.conv2, nn.ReLU(), self.pool, self.dropout)
        self.FCnet1 = nn.Sequential(self.fc1, nn.ReLU(), self.fc2)
        self.FCnet2 = nn.Sequential(self.fc1, nn.ReLU(), self.fc2)


    def forward(self, x):
        if self.weight_sharing_CNN == True:
            y1 = self.CNNnet1(torch.unsqueeze(x[:,0],dim=1))
            y2 = self.CNNnet1(torch.unsqueeze(x[:,1],dim=1))

        else:
            y1 = self.CNNnet1(torch.unsqueeze(x[:,0],dim=1))
            y2 = self.CNNnet2(torch.unsqueeze(x[:,1],dim=1))

        y1 = y1.view(-1, 64 *2 *2)
        y2 = y2.view(-1, 64 *2 *2)
        y = torch.cat((y1, y2), dim = 1)
        if self.weight_sharing_FC == True:
            y1 = self.FCnet1(y1)
            #out1 = F.softmax(y1)
            y2 = self.FCnet1(y2)
            #out2 = F.softmax(y2)

        else:
            y1 = self.FCnet1(y1)
          #  out1 = F.softmax(y1)
            y2 = self.FCnet2(y2)
           # out2 = F.softmax(y2)


        y = y.view(-1, 64*2*2*2)
        y = self.fc3(y)
        y = self.fc4(y)
        y = self.fc5(y)
        #y=self.fc6(y)
        if self.auxiliary_loss == True:
            return y1, y2, y
        else:
            return y
