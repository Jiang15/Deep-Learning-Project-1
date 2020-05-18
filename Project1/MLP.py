class MLP(nn.Module):
    def __init__(self, input_channels, output_channels, weight_sharing, auxiliary_loss):

        super(MLP).__init__()
        self.weight_sharing = weight_sharing
        self.auxiliary_loss = auxiliary_loss
        self.fc4 = nn.Linear(20, 2)


        self.FCnet1 = nn.Sequential(nn.Linear(14 * 14, 128),
                                    nn.Linear(128, 64),
                                    nn.Linear(64, nb_digits))
        self.FCnet2 = nn.Sequential(nn.Linear(14 * 14, 128),
                                    nn.Linear(128, 64),
                                    nn.Linear(64, nb_digits)
                                    )

    def forward(self, x):
        # print(x.shape)#2,2,14,14
        # print(x.shape[0])
        _x1 = torch.reshape(x[:, 0, :, :], (-1, 1, 14, 14))
        _x1 = torch.reshape(_x1, (_x1.shape[0], -1))
        _x2 = torch.reshape(x[:, 1, :, :], (-1, 1, 14, 14))
        _x2 = torch.reshape(_x2, (_x2.shape[0], -1))
        # _x3=torch.reshape(x,(x.shape[0],-1))
        # print(_x2.shape)

        if self.weight_sharing == True:
            y1 = self.FCnet1(_x1)
            y2 = self.FCnet1(_x2)
        else:
            y1 = self.FCnet1(_x1)
            y2 = self.FCnet2(_x2)
        y = torch.cat((y1, y2), 1)
        y = self.fc4(y)
        if self.auxiliary_loss == True:
            return y1, y2, y
        else:
            return y