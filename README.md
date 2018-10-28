# MonoDepth
![demo.gif animation](readme_images/demo.gif)

This repo is inspired by an amazing work of [Clément Godard](http://www0.cs.ucl.ac.uk/staff/C.Godard/), [Oisin Mac Aodha](http://vision.caltech.edu/~macaodha/) and [Gabriel J. Brostow](http://www0.cs.ucl.ac.uk/staff/g.brostow/) for Unsupervised Monocular Depth Estimation.
Original code and paper could be found via the following links:
1. [Original repo](https://github.com/mrharicot/monodepth)
2. [Original paper](https://arxiv.org/abs/1609.03677)

## MonoDepth-PyTorch
This repository contains code and additional parts for the PyTorch port of the MonoDepth Deep Learning algorithm. For more information about original work, please visit [author's website](http://visual.cs.ucl.ac.uk/pubs/monoDepth/)

## Purpose

Purpose of this repository is to make a more lightweight model for depth estimation with better accuracy.
In our version of MonoDepth, we used ResNet50 as an encoder. It was slightly changed (with one more lateral shrinkage) as well as in the original repo.

Also, we add ResNet18 version and used batch normalization in both cases for training stability.
Moreover, we made flexible feature extractor with any version of original Resnet from torchvision models zoo
 with an option to use pretrained models.

## Dataset
### KITTI

This algorithm requires stereo-pair images for training and single images for testing.
[KITTI](http://www.cvlibs.net/datasets/kitti/raw_data.php) dataset was used for training.
It contains 38237 training samples.
Raw dataset (about 175 GB) can be downloaded by running:
```shell
wget -i kitti_archives_to_download.txt -P ~/my/output/folder/
```
kitti_archives_to_download.txt may be found in this repo.

## Dataloader
Dataloader assumes the following structure of the folder with train examples (**'data_dir'** argument contains path to that folder):
The folder contains subfolders with following folders "image_02/data" for left images and  "image_03/data" for right images.
Such structure is default for KITTI dataset

Example data folder structure (path to the "kitti" directory should be passed as **'data_dir'** in this example):
```
data
├── kitti
│   ├── 2011_09_26_drive_0001_sync
│   │   ├── image_02
│   │   │   ├─ data
│   │   │   │   ├── 0000000000.png
│   │   │   │   └── ...
│   │   ├── image_03
│   │   │   ├── data
│   │   │   │   ├── 0000000000.png
│   │   │   │   └── ...
│   ├── ...
├── models
├── output
├── test
│   ├── left
│   │   ├── test_1.jpg
│   │   └── ...
```
    
## Training
Example of training can be find in [Monodepth](Monodepth.ipynb) notebook.

Model class from main_monodepth_pytorch.py should be initialized with following params (as easydict) for training:
 - `data_dir`: path to the dataset folder
 - `val_data_dir`: path to the validation dataset folder
 - `model_path`: path to save the trained model
 - `output_directory`: where save dispairities for tested images
 - `input_height`
 - `input_width`
 - `model`: model for encoder (resnet18_md or resnet50_md or any torchvision version of Resnet (resnet18, resnet34 etc.)
 - `pretrained`: if use a torchvision model it's possible to download weights for pretrained model
 - `mode`: train or test
 - `epochs`: number of epochs,
 - `learning_rate`
 - `batch_size`
 - `adjust_lr`: apply learning rate decay or not
 - `tensor_type`:'torch.cuda.FloatTensor' or 'torch.FloatTensor'
 - `do_augmentation`:do data augmentation or not
 - `augment_parameters`:lowest and highest values for gamma, lightness and color respectively
 - `print_images`
 - `print_weights`
 - `input_channels` Number of channels in input tensor (3 for RGB images)
 - `num_workers` Number of workers to use in dataloader

Optionally after initialization, we can load a pretrained model via `model.load`.

After that calling train() on Model class object starts the training process.

Also, it can be started via calling main_monodepth_pytorch.py through the terminal and feeding parameters as argparse arguments.

## Train results and pretrained model

Results presented on the gif image were obtained using the model with a **resnet18** as an encoder, which can be downloaded from [here](https://my.pcloud.com/publink/show?code=XZb5r97ZD7HDDlc237BMjoCbWJVYMm0FLKcy).

For training the following parameters were used:
```
`model`: 'resnet18_md'
`epochs`: 200,
`learning_rate`: 1e-4,
`batch_size`: 8,
`adjust_lr`: True,
`do_augmentation`: True
```
The provided model was trained on the whole dataset, except subsets, listed below, which were used for a hold-out validation.

```
2011_09_26_drive_0002_sync  2011_09_29_drive_0071_sync
2011_09_26_drive_0014_sync  2011_09_30_drive_0033_sync
2011_09_26_drive_0020_sync  2011_10_03_drive_0042_sync
2011_09_26_drive_0079_sync
```

The demo gif image is a visualization of the predictions on `2011_09_26_drive_0014_sync` subset.

See [Monodepth](Monodepth.ipynb) notebook for the details on the training.
    
## Testing
Example of testing can also be find in [Monodepth](Monodepth.ipynb) notebook.

Model class from main_monodepth_pytorch.py should be initialized with following params (as easydict) for testing:
 - `data_dir`: path to the dataset folder
 - `model_path`: path to save the trained model
 - `pretrained`: 
 - `output_directory`: where save dispairities for tested images
 - `input_height`
 - `input_width`
 - `model`: model for encoder (resnet18 or resnet50)
 - `mode`: train or test
 - `input_channels` Number of channels in input tensor (3 for RGB images)
 - `num_workers` Number of workers to use in dataloader
 
After that calling test() on Model class object starts testing process.

Also it can be started via calling [main_monodepth_pytorch.py](main_monodepth_pytorch.py) through the terminal and feeding parameters as argparse arguments. 
    
## Requirements
This code was tested with PyTorch 0.4.1, CUDA 9.1 and Ubuntu 16.04. Other required modules:

```
torchvision
numpy
matplotlib
easydict
```
