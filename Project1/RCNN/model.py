import torch
from torch import nn
import torch.nn.functional as F




class RCL(nn.Module):
    def __init__(self,weight_sharing_recurr, K, steps = 2):
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



class RCNN(nn.Module):
    def __init__(self, channels, num_classes, weight_sharing_recurr, auxiliary_loss, K = 32, steps = 2):
        super(RCNN, self).__init__()
        self.weight_sharing_recurr = weight_sharing_recurr
        self.auxiliary_loss = auxiliary_loss
        self.K = K

        self.layer1 = nn.Conv2d(channels, K, kernel_size = 3, padding = 1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(K)
        self.pooling = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer2 = RCL(weight_sharing_recurr, K, steps=steps)
        self.layer3 = RCL(weight_sharing_recurr, K, steps=steps)
        self.fc = nn.Linear(K, num_classes, bias = True)
        self.dropout = nn.Dropout(p=0.3)

        self.layer1_aux = nn.Conv2d(1, K, kernel_size = 3, padding = 1)
        self.FC_aux = nn.Linear(self.K * 14 * 14, 10)

    def forward(self, x):
        input = x
        x = self.bn(self.relu(self.layer1(x)))
        x = self.pooling(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.dropout(x)
        x = self.layer3(x)
        x = F.max_pool2d(x, x.shape[-1])
        x = x.view(-1, self.K)
        x = self.dropout(x)
        x = self.fc(x)
        if self.auxiliary_loss:
            y1 = torch.unsqueeze(input[:,0],dim=1)
            y2 = torch.unsqueeze(input[:,1],dim=1)
            y1 = self.bn(self.relu(self.layer1_aux(y1)))
            y2 = self.bn(self.relu(self.layer1_aux(y2)))
            y1 = self.layer2(y1)
            y2 = self.layer2(y2)
            y1 = y1.view(-1, self.K * 14 * 14)
            y2 = y2.view(-1, self.K * 14 * 14)
            y1 = self.FC_aux(y1)
            y2 = self.FC_aux(y2)
            return y1, y2, x
        else:
            return x
