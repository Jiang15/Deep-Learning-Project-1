import torch
from torch import nn
import torch.nn.functional as F


class RCL(nn.Module):
    def __init__(self,weight_sharing_recurr, K, steps):
        super(RCL, self).__init__()
        self.weight_sharing_recurr = weight_sharing_recurr
        self.conv = nn.Conv2d(K, K, kernel_size=3, stride=1, padding=1, bias=False)
        self.convList = nn.ModuleList([nn.Conv2d(K, K, kernel_size=3, stride=1, padding=1, bias=False) for i in range(steps)])
        self.bn = nn.ModuleList([nn.BatchNorm2d(K) for i in range(steps)])
        self.relu = nn.ReLU(inplace=True)
        self.steps = steps
        self.recurr = nn.Conv2d(K, K, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        rx = x
        for i in range(self.steps):
            if i == 0:
                x = self.conv(x)
            else:
                if self.weight_sharing_recurr:
                    x = self.conv(x) + self.recurr(rx)
                else:
                    x = self.convList[i](x)
            x = self.relu(x)
            x = self.bn[i](x)
        return x


class CNN(nn.Module):
    def __init__(self, channels, num_classes, weight_sharing_recurr, auxiliary_loss, K = 32, steps = 2):
        super(CNN, self).__init__()
        self.weight_sharing_recurr = weight_sharing_recurr
        self.auxiliary_loss = auxiliary_loss
        self.K = K

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(K)
        self.pooling = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

        self.layer1 = nn.Conv2d(1, K, kernel_size = 3, padding = 1)
        self.layer2 = RCL(weight_sharing_recurr, K, steps=steps)
        self.layer3 = RCL(weight_sharing_recurr, K, steps=steps)
        self.fc = nn.Linear(K, num_classes, bias = True)
        self.dropout = nn.Dropout(p=0.3)
        self.FC_aux = nn.Linear(self.K * 7 * 7, 10)

    def forward(self, x):
        x1 = torch.unsqueeze(x[:,0],dim=1)
        x2 = torch.unsqueeze(x[:,1],dim=1)
        x1 = self.bn(self.relu(self.layer1(x1)))
        x2 = self.bn(self.relu(self.layer1(x2)))
        x1 = self.pooling(x1)
        x2 = self.pooling(x2)
        x1 = self.dropout(x1)
        x2 = self.dropout(x2)
        x1 = self.layer2(x1)
        x1 = self.layer2(x1)
        x1 = self.dropout(x1)
        x2 = self.dropout(x2)
        x = x1 + x2
        x = self.layer3(x)
        x = F.max_pool2d(x, x.shape[-1])
        x = x.view(-1, self.K)
        x = self.dropout(x)
        x = self.fc(x)
        if self.auxiliary_loss:
            y1 = x1.view(-1, self.K * 7 * 7)
            y2 = x2.view(-1, self.K * 7 * 7)
            y1 = self.FC_aux(y1)
            y2 = self.FC_aux(y2)
            return y1, y2, x
        else:
            return x

class CNN2(nn.Module):
    def __init__(self, channels, num_classes, weight_sharing_recurr, auxiliary_loss, K = 32, steps = 2):
        super(CNN2, self).__init__()
        self.weight_sharing_recurr = weight_sharing_recurr
        self.auxiliary_loss = auxiliary_loss
        self.K = K
        self.steps = steps
        self.layer1 = nn.Conv2d(1, K, kernel_size = 3, padding = 1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(K)
        self.pooling = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer2 = RCL(weight_sharing_recurr, K, steps=steps)
        self.layer3 = RCL(weight_sharing_recurr, K, steps=steps)
        self.fc = nn.Linear(K, num_classes, bias = True)
        self.dropout = nn.Dropout(p=0.3)
        self.FC_aux = nn.Linear(self.K * 7 * 7, 10)
        self.convList = nn.ModuleList([nn.Conv2d(K, K, kernel_size=3, stride=1, padding=1, bias=False) for i in range(steps)])
        self.bnList = nn.ModuleList([nn.BatchNorm2d(K) for i in range(steps)])

    def forward(self, x):
        x1 = torch.unsqueeze(x[:,0],dim=1)
        x2 = torch.unsqueeze(x[:,1],dim=1)
        x1 = self.bn(self.relu(self.layer1(x1)))
        x2 = self.bn(self.relu(self.layer1(x2)))
        x1 = self.pooling(x1)
        x2 = self.pooling(x2)
        x1 = self.dropout(x1)
        x2 = self.dropout(x2)
        if self.weight_sharing_recurr:
            x1 = self.layer2(x1)
            x1 = self.layer2(x1)
            x1 = self.dropout(x1)
            x2 = self.dropout(x2)
        else:
            for i in range(self.steps):
                x1 = self.relu(self.convList[i](x1))
                x1 = self.bnList[i](x1)
                x2 = self.relu(self.convList[i](x2))
                x2 = self.bnList[i](x2)
                x1 = self.dropout(x1)
                x2 = self.dropout(x2)
        x = x1 + x2
        x = self.layer3(x)
        x = F.max_pool2d(x, x.shape[-1])
        x = x.view(-1, self.K)
        x = self.dropout(x)
        x = self.fc(x)
        if self.auxiliary_loss:
            y1 = x1.view(-1, self.K * 7 * 7)
            y2 = x2.view(-1, self.K * 7 * 7)
            y1 = self.FC_aux(y1)
            y2 = self.FC_aux(y2)
            return y1, y2, x
        else:
            return x




class RCNN2(nn.Module):
    def __init__(self, channels, num_classes, weight_sharing_recurr, auxiliary_loss, K = 32, steps = 2):
        super(RCNN2, self).__init__()
        self.weight_sharing_recurr = weight_sharing_recurr
        self.auxiliary_loss = auxiliary_loss
        self.K = K
        self.layer1 = nn.Conv2d(1, K, kernel_size = 3, padding = 1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(K)
        self.pooling = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer2 = RCL(weight_sharing_recurr, K, steps=steps)
        self.layer3 = RCL(weight_sharing_recurr, K, steps=steps)
        self.fc = nn.Linear(K, num_classes, bias = True)
        self.dropout = nn.Dropout(p=0.3)
        self.FC_aux = nn.Linear(self.K * 7 * 7, 10)

    def forward(self, x):
        x1 = torch.unsqueeze(x[:,0],dim=1)
        x2 = torch.unsqueeze(x[:,1],dim=1)
        x1 = self.bn(self.relu(self.layer1(x1)))
        x2 = self.bn(self.relu(self.layer1(x2)))
        x1 = self.pooling(x1)
        x2 = self.pooling(x2)
        x1 = self.dropout(x1)
        x2 = self.dropout(x2)
        x1 = self.layer2(x1)
        x1 = self.layer2(x1)
        x1 = self.dropout(x1)
        x2 = self.dropout(x2)
        x = x1 + x2
        x = self.layer3(x)
        x = F.max_pool2d(x, x.shape[-1])
        x = x.view(-1, self.K)
        x = self.dropout(x)
        x = self.fc(x)
        if self.auxiliary_loss:
            y1 = x1.view(-1, self.K * 7 * 7)
            y2 = x2.view(-1, self.K * 7 * 7)
            y1 = self.FC_aux(y1)
            y2 = self.FC_aux(y2)
            return y1, y2, x
        else:
            return x
