from Project2.dataset import generate_disc_set
from Project2.layers import Linear, ReLU
from Project2.loss_func import MSELoss
from Project2.optimizers import SGD
from Project2.Sequential import Sequential
from matplotlib import pyplot as plt

train_input, train_label = generate_disc_set(1000)
test_input, test_label = generate_disc_set(1000)

def train(model, criterion, optimizer, input, target, nb_epochs):
    loss_history = []
    model.zero_grad()
    for e in range(nb_epochs):
        loss_e = 0.
        for b in range(0, input.shape[0]):
            output = model.forward(input.narrow(0, b, 1))
            loss = criterion.forward(output, target.narrow(0, b, 1))
            loss_e += loss.item()
            loss_history.append(loss_e)
            tmp = criterion.backward(output, target.narrow(0, b, 1))
            model.backward(tmp)
            optimizer.step()
    return loss_history

model = Sequential([Linear(2,25),ReLU(),Linear(25,50),ReLU(),Linear(50,25), ReLU(),Linear(25,1)])
criterion = MSELoss()
optimizer = SGD(parameters = model.param(), lr = 1e-1)
loss_train=train(model,criterion,optimizer,train_input,train_label, nb_epochs = 50)
plt.plot(loss_train)

