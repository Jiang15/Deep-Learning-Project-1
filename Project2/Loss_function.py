import _ini_

class MSELoss(object):
    def __init__(self):
        super(MSELoss, self).__init__()

    def loss(self, y, y_pred):
        y = y.view(y_pred.size())
        loss = (y - y_pred).pow(2).sum()
        return loss

    def grad(self, y, y_pred):
        y = y.view(y_pred.size())
        grad = 2 * (y - y_pred)
        return grad


class BCELoss(object):
    def __init__(self):
        super(BCELoss, self).__init__()

    def loss(self, y, y_pred):
        y = y.view(y_pred.size())
        loss = -y_pred * torch.log(y) - (1 - y_pred) * torch.log(1 - torch.sigmoid(y))
        return loss

    def grad(self, y, y_pred):
        grad = -y_pred / y - (1 - y_pred) * (-1 / (1 - torch.sigmoid(y))) * (torch.exp(-y) / (1 + torch.exp(-y)) ** 2)
        return grad