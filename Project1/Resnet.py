import torch
import torch.nn.functional as F
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, kernel_size):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(12, 12, kernel_size, padding=(kernel_size - 1) // 2)
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(12, 12, kernel_size, padding=(kernel_size - 1) // 2)
        self.bn2 = nn.BatchNorm2d(12)

    def forward(self, x):
        y = self.bn1(self.conv1(x))
        y = F.relu(y)
        y = self.bn2(self.conv2(y))
        y += x
        y = F.relu(y)
        return y


class ResNet(nn.Module):
    def __init__(self, kernel_size, nb_blocks, weight_sharing, auxiliary_loss):
        super(ResNet, self).__init__()
        self.weight_sharing = weight_sharing
        self.auxiliary_loss = auxiliary_loss
        self.conv0 = nn.Conv2d(1, 12, kernel_size=1)
        self.resblocks1 = nn.Sequential(*(ResBlock(kernel_size) for _ in range(nb_blocks)))
        self.resblocks2 = nn.Sequential(*(ResBlock(kernel_size) for _ in range(nb_blocks)))
        self.avg = nn.AvgPool2d(kernel_size=2)
        self.fc1 = nn.Linear(12 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 10)
        self.fc4 = nn.Linear(12 * 7 * 7 * 2, 128)
        self.fc5 = nn.Linear(128, 32)
        self.fc6 = nn.Linear(32, 2)

        self.fcnet1 = nn.Sequential(self.fc1, self.fc2, self.fc3)
        self.fcnet2 = nn.Sequential(self.fc1, self.fc2, self.fc3)
        self.fcnet3 = nn.Sequential(self.fc4, self.fc5, self.fc6)

    def forward(self, x):
        x1 = torch.reshape(x[:, 0, :, :], (-1, 1, 14, 14))
        x2 = torch.reshape(x[:, 1, :, :], (-1, 1, 14, 14))
        print(x1.shape)
        print(x2.shape)
        x1 = self.conv0(x1)
        x2 = self.conv0(x2)
        if self.weight_sharing == True:
            y1 = self.resblocks1(x1)
            y2 = self.resblocks1(x2)
        else:
            y1 = self.resblocks1(x1)
            y2 = self.resblocks2(x2)

        y1 = F.relu(self.avg(y1))
        y2 = F.relu(self.avg(y2))
        print(y1.shape)
        print(y2.shape)
        # yl_1=self.fcnet1(y1)
        # yl_2=self.fcnet1(y2)
        y_1 = y1.view(-1, 12 * 7 * 7)
        y_2 = y2.view(-1, 12 * 7 * 7)
        y = torch.cat((y_1, y_2), 1)
        y_1 = self.fcnet1(y_1)
        y_2 = self.fcnet2(y_2)
        y = y.view(-1, 12 * 7 * 7 * 2)
        y = self.fcnet3(y)
        if self.auxiliary_loss == True:
            return y_1, y_2, y
        else:
            return y