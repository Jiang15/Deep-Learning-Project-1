import math
from torch import empty,set_grad_enabled
from matplotlib import pyplot as plt



set_grad_enabled(False)


def generate_disc_set(nb):
    pts = empty(nb, 2).uniform_(0, 1)
    target = (-((pts-0.5).pow(2).sum(1)-(1 / (2 * math.pi))).sign()+1)/2 # take reversed sign of subtraction with 2/pi then add 1 and divide by 2 then reverse
    return pts, target


train_input, train_label = generate_disc_set(1000)
test_input, test_label = generate_disc_set(1000)

num_0 = train_input[train_label == 0]
num_1 = train_input[train_label == 1]

plt.scatter(num_0[:, 0], num_0[:, 1], label='class 0', color='blue')
plt.scatter(num_1[:, 0], num_1[:, 1], label='class 1', color='red')
plt.legend()
plt.xlabel('x position of data points')
plt.ylabel('y position of data points')
plt.title('Datapoints')
plt.show()

