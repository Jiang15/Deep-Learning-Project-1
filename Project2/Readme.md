# Weight sharing and auxiliary losses for classification
| Student's name | SCIPER |
| -------------- | ------ |
| Wei Jiang | 313794  |
| Minghui Shi | 308209 |
| Xiaoqi Ma | |

## Project Description
The objective of this project is to design a mini “deep learning framework” using only pytorch’s
tensor operations and the standard math library, hence in particular without using autograd or the
neural-network modules.


### Folders and Files
- `model`:
  - `layers.py`: contains Linear for fully connected layer, Relu, Tanh, Leaky Relu, Exponential Relu and Sigmoid activation function layers.
  - `loss_func.py`: contains MSE loss function
  - `optimizers.py`: condtains SGD, moment SGD, Adagrad, Adam optmizers
  - `Module.py`: condtains model for layers
  - `Sequential.py`: condtains Sequential function to connect different layers
- `helpers.py`: contains helper functions for training and testing
- `test.py`: run to train and test model with three hidden layers of 25 units using SGD as optimizer, MES loss as loss function and Relu, Tanh as activation function.  


  
## Getting Started
- Run `test.py` to train and test the model with 50 epochs. Cross validation is automatically run for get best learning rate. The MSE loss, training and testing error is logging for each epoch. Final training and testing error is printed at the end.
