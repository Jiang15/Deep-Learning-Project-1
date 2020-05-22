# Weight sharing and auxiliary losses for classification
| Student's name | SCIPER |
| -------------- | ------ |
| Wei Jiang | 313794  |
| Minghui Shi | 308209 |
| Xiaoqi Ma | |

## Project Description
The objective of this project is to test different architectures to compare two digits visible in a two-channel image. It aims at showing in particular the impact of weight sharing, and of the use of an auxiliary loss to help the training of the main objective.


### Folders and Files
  - `metric.py`: contains evaluation metric for quality measurement
  - `train.py`: run this file to train the model on GPU
  - `test.py`: run this file to apply the trained model to given image files in `testset\` and save the generated mask in `test_mask.tif`
  - `validation.py`: condtains evaluation function for validation process
  - `util_pred(_cpu).py`: condtains some utility function for prediction.
  - `tile_helpers(_cpu).py`: condtains tile function and untile function.
  - `segment`: contains watershed function for labeling
  - `model.pkl`: contains models generated by `train.py`
  - `dataset.py`: contains all the dataset for training, validation.
  - `unet.py`: contains structure of U-net model.
  - `unet_part.py`: contains some detail implementation in U-net model.
  - `split_image.py`: split image for a input image series to single images
If you want to run train.py please add data under './data/frame/' and './data/mask/' respectively. For example, './data/frame/Denis_Joly_ADH1_27.1_frames.tif'
  
## Getting Started
- Run `test.py` to get the mask of input frame

