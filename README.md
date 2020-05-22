# Weight sharing and auxiliary losses for classification
| Student's name | SCIPER |
| -------------- | ------ |
| Wei Jiang | 313794  |
| Minghui Shi | 308209 |
| Xiaoqi Ma | |

## Project Description
The objective of this project is to test different architectures to compare two digits visible in a two-channel image. It aims at showing in particular the impact of weight sharing, and of the use of an auxiliary loss to help the training of the main objective.


### Folders and Files
- `models`:
  - `FNN.py`: contains FNN model
  - `RCNN.py`: contains RCNN model
  - `ResNet.py`: condtains ResNet model
  - `Siamese.py`: condtains SiameseNet model
- `helpers.py`: contains helper functions for training and testing
- `test.py`: run to traininng the four models. It trains each models for 11 rounds and 25 epochs in each round.  


  
## Getting Started
- Run `test.py` to train the four models. Mean loss and accucy and its standard deviation will print at the end for three cases: simple structure, structure with weight sharing and auxiliary loss.  

If you want to test cross validation to check learning rate for four models, please reset the boolean flag - 'run_cross_validation' varaible to true. 

