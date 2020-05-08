import torch
from torch import nn
import torch.nn.functional as F


class RCL(nn.Module):
    def __init__(self, K, steps):
        super(RCL, self).__init__()
        self.conv = nn.Conv2d(K, K, kernel_size=3, stride=1, padding=1, bias=False)
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
                rx = self.recurr(rx)
                x = self.conv(x) + rx
            x = self.relu(x)
            x = self.bn[i](x)
        return x

class RCL_recurr(nn.Module):
    def __init__(self,weight_sharing_recurr, K, steps):
        super(RCL_recurr, self).__init__()
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
                    rx = self.recurr(rx) #weight sharing in recurrent connection
                else:
                    rx = self.convList[i](rx)     # not share in recurrent
                x = self.conv(x) + rx
            x = self.relu(x)
            x = self.bn[i](x)
        return x


class CNN(nn.Module):
    def __init__(self, channels, num_classes, weight_sharing, auxiliary_loss, K = 32, steps = 3):
        super(CNN, self).__init__()
        self.weight_sharing = weight_sharing
        self.auxiliary_loss = auxiliary_loss
        self.K = K
        self.steps = steps

        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(K)
        self.bn2 = nn.BatchNorm2d(K)

        self.pooling = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

        self.convList = nn.ModuleList([nn.Conv2d(K, K, kernel_size=3, stride=1, padding=1, bias=False) for i in range(steps)])
        self.bnList = nn.ModuleList([nn.BatchNorm2d(K) for i in range(steps)])

        self.layer1 = nn.Conv2d(1, K, kernel_size = 3, padding = 1)
        self.layer2 = nn.Conv2d(1, K, kernel_size = 3, padding = 1)
        self.rcl1 = RCL(self.K, steps=self.steps)
        self.rcl2 = RCL(self.K, steps=self.steps)
        self.rcl3 = RCL(self.K, steps=self.steps)

        self.fc1 = nn.Linear(K * 7 * 7, 40, bias = True)
        self.fc2 = nn.Linear(40, num_classes, bias = True)
        self.dropout = nn.Dropout(p=0.3)
        self.fc_aux = nn.Linear(K * 7 * 7, 10)

    def forward(self, x):
        x1 = torch.unsqueeze(x[:,0],dim=1)
        x2 = torch.unsqueeze(x[:,1],dim=1)
        x1 = self.bn1(self.relu(self.layer1(x1)))
        x2 = self.bn2(self.relu(self.layer2(x2)))
        x1 = self.pooling(x1)
        x2 = self.pooling(x2)
        x1 = self.dropout(x1)
        x2 = self.dropout(x2)
        if self.weight_sharing:
            x1 = self.rcl1(x1)
            x2 = self.rcl2(x2)
        else:
            for i in range(self.steps):
                x1 = self.convList[i](x1)
                x2 = self.convList[i](x2)
                x1 = self.relu(x1)
                x2 = self.relu(x2)
                x1 = self.bnList[i](x1)
                x2 = self.bnList[i](x2)
        x1 = self.dropout(x1)
        x2 = self.dropout(x2)
        x = x1 + x2
        if self.weight_sharing:
            x = self.rcl3(x)
        else:
            for i in range(self.steps):
                x = self.convList[i](x)
                x = self.relu(x)
                x = self.bnList[i](x)
        # x = F.max_pool2d(x, x.shape[-1])
        x = x.view(-1, self.K * 7 * 7)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.fc2(x)

        if self.auxiliary_loss:
            y1 = x1.view(-1, self.K * 7 * 7)
            y2 = x2.view(-1, self.K * 7 * 7)
            y1 = self.fc_aux(y1)
            y2 = self.fc_aux(y2)
            return y1, y2, x
        else:
            return x

class CNN_recurr(nn.Module):
    def __init__(self, channels, num_classes, weight_sharing_recurr, auxiliary_loss, K = 32, steps = 3):
        super(CNN_recurr, self).__init__()
        self.weight_sharing_recurr = weight_sharing_recurr
        self.auxiliary_loss = auxiliary_loss
        self.K = K
        self.steps = steps

        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(K)
        self.bn2 = nn.BatchNorm2d(K)

        self.pooling = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

        self.convList = nn.ModuleList([nn.Conv2d(K, K, kernel_size=3, stride=1, padding=1, bias=False) for i in range(steps)])
        self.bnList = nn.ModuleList([nn.BatchNorm2d(K) for i in range(steps)])
        # self.rclList = nn.ModuleList([RCL_recurr(weight_sharing_recurr, K, steps=steps) for i in range(steps)])

        self.layer1 = nn.Conv2d(1, K, kernel_size = 3, padding = 1)
        self.layer2 = nn.Conv2d(1, K, kernel_size = 3, padding = 1)

        self.rcl1 = RCL_recurr(weight_sharing_recurr, K, steps=steps)
        self.rcl2 = RCL_recurr(weight_sharing_recurr, K, steps=steps)
        self.rcl3 = RCL_recurr(weight_sharing_recurr, K, steps=steps)

        self.fc1 = nn.Linear(K * 7 * 7, 40, bias = True)
        self.fc2 = nn.Linear(40, num_classes, bias = True)
        self.dropout = nn.Dropout(p=0.3)
        self.fc_aux = nn.Linear(K * 7 * 7, 10)

    def forward(self, x):
        x1 = torch.unsqueeze(x[:,0],dim=1)
        x2 = torch.unsqueeze(x[:,1],dim=1)
        x1 = self.bn1(self.relu(self.layer1(x1)))
        x2 = self.bn2(self.relu(self.layer2(x2)))
        x1 = self.pooling(x1)
        x2 = self.pooling(x2)
        x1 = self.dropout(x1)
        x2 = self.dropout(x2)
        x1 = self.rcl1(x1)
        x2 = self.rcl2(x2)
        x1 = self.dropout(x1)
        x2 = self.dropout(x2)
        x = x1 + x2
        x = self.rcl3(x)
        # x = F.max_pool2d(x, x.shape[-1])
        x = x.view(-1, self.K * 7 * 7)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.fc2(x)

        if self.auxiliary_loss:
            y1 = x1.view(-1, self.K * 7 * 7)
            y2 = x2.view(-1, self.K * 7 * 7)
            y1 = self.fc_aux(y1)
            y2 = self.fc_aux(y2)
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
