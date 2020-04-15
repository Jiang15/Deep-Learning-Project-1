import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import Tensor as tensor
from dlc_practical_prologue import generate_pair_sets

from torchsummary import summary

import matplotlib.pyplot as plt

import math


class CNN(nn.Module):
    def __init__(self, input_channels, output_channels, auxiliary_loss):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 32, 3)
        self.conv1_aux = nn.Conv2d(1, 32, 3)

        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(32, 64, 3) # auxiliary loss flows through this layer of primary objective
        self.fc1 = nn.Linear(64 * 2 * 2, 50) # auxiliary loss flows through this layer of primary objective
        self.fc1_aux = nn.Linear(64 * 2 * 2, 50)
        self.fc2 = nn.Linear(50, 2) #output
        self.fc2_aux = nn.Linear(50, 10)#output

        self.subCNN = nn.Sequential(self.conv1, nn.ReLU(), self.pool, self.conv2, nn.ReLU(), self.pool)
        self.subCNN_aux = nn.Sequential(self.conv1_aux, nn.ReLU(), self.pool, self.conv2, nn.ReLU(), self.pool)

        self.FC = nn.Sequential(self.fc1, nn.ReLU(), self.fc2)
        self.FC_aux = nn.Sequential(self.fc1, nn.ReLU(), self.fc2_aux)

        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        input =  x
        x = self.dropout(self.subCNN(x))
        x = x.view(-1, 64 * 2 * 2)
        output = self.FC(x)

        if auxiliary_loss:
            y1 = torch.unsqueeze(input[:,0],dim=1)
            y2 = torch.unsqueeze(input[:,1],dim=1)
            y1 = self.dropout(self.subCNN_aux(y1))
            y2 = self.dropout(self.subCNN_aux(y2))
            y1 = y1.view(-1, 64 * 2 * 2)
            y2 = y2.view(-1, 64 * 2 * 2)
            y1 = self.FC_aux(y1)
            y2 = self.FC_aux(y2)
            return y1, y2, output
        else:
            return output


