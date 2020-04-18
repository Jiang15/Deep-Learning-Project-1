import math
import torch
from matplotlib import pyplot as plt

train_input = torch.empty(1000, 2).uniform_(0, 1)
test_input = torch.empty(1000, 2).uniform_(0, 1)

train_label = ((train_input[:, 0] - 0.5).pow(2) + (train_input[:, 1] - 0.5).pow(2)).pow(0.5) > 1 / math.sqrt(
    2 * math.pi)
test_label = ((test_input[:, 0] - 0.5).pow(2) + (test_input[:, 1] - 0.5).pow(2)).pow(0.5) > 1 / math.sqrt(2 * math.pi)

num_0 = train_input[train_label == 0]
num_1 = train_input[train_label == 1]

plt.scatter(num_0[:, 0], num_0[:, 1], label='class 0', color='red')
plt.scatter(num_1[:, 0], num_1[:, 1], label='class 1', color='blue')
plt.legend()
plt.xlabel('x position of data points')
plt.ylabel('y position of data points')
plt.title('Datapoints')
plt.show()
