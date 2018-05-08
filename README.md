# PyTorch Transfer Learning

PyTorch transfer learning example developed as part of Udacity's AI Programming with Python Nanodegree program.

## Getting Started
### Environment
Tested on the following environment:
- Ubuntu 16.04
- Python3:
    - Numpy
    - PyTorch
    - TorchVision

- NVIDAI GPU (driver version 390.48)
- CUDA 9.1

### Sample Data
Download sample data using curl:

```
curl -O https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz
```

And extract using tar:
```
mkdir flowers
tar -xvzf flower_data.tar.gz -C flowers
```

## Training
### Usage
```
python train.py [-h] [--save_dir SAVE_DIR]
                [--arch {vgg13_bn,vgg16_bn,densenet161,resnet50,resnet18,densenet201,densenet121,vgg19_bn,resnet34}]
                [--learning_rate LEARNING_RATE] [--hidden_units HIDDEN_UNITS]
                [--epochs EPOCHS] [--gpu] [--num_workers NUM_WORKERS]
                [--pin_memory]
                data_dir

positional arguments:
  data_dir              Directory used to locate source images

optional arguments:
  -h, --help            show this help message and exit
  --save_dir SAVE_DIR   Directory used to save checkpoints
  --arch {vgg13_bn,vgg16_bn,densenet161,resnet50,resnet18,densenet201,densenet121,vgg19_bn,resnet34}
                        Model architecture to use for training
  --learning_rate LEARNING_RATE
                        Learning rate hyperparameter
  --hidden_units HIDDEN_UNITS
                        Number of hidden units hyperparameter
  --epochs EPOCHS       Number of epochs used to train model
  --gpu                 Use GPU for training
  --num_workers NUM_WORKERS
                        Number of subprocesses to use for data loading
  --pin_memory          Request data loader to copy tensors into CUDA pinned
                        memory
```
### Model Architectures
The following model architectures are available:

| Model | Reference |
| --- | --- |
| VGG | [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556) |
| DenseNet | [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993) |
| ResNet | [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) |

### Examples
The following will train a densenet121 model on the GPU for 5 epochs:

```
python train.py flowers --arch densenet121 --epochs 5 --gpu --pin_memory --num_workers 4
```

The following will train a vgg13 model on the GPU for 5 epochs:

```
python train.py flowers --arch vgg13_bn --epochs 5 --gpu --pin_memory --num_workers 4
```

The following will train a resnet18 model on the CPU for 3 epochs and save the checkpoint in the checkpoints directory:

```
python train.py flowers --arch resnet18 --epochs 3 --save_dir checkpoints
```

## Inference
### Usage
```
python predict.py [-h] [--top_k TOP_K] [--category_names CATEGORY_NAMES]
                  [--gpu] [--verbose]
                  input checkpoint

positional arguments:
  input                 Input image
  checkpoint            Model checkpoint file to use for prediction

optional arguments:
  -h, --help            show this help message and exit
  --top_k TOP_K         Return top k most likely classes
  --category_names CATEGORY_NAMES
                        Mapping file used to map categories to real names
  --gpu                 Use GPU for prediction
  --verbose             Display additional processing information
```
### Examples
The following will return the most likely class using a densenet121 checkpoint executing on the GPU:

```
python predict.py flowers/test/28/image_05230.jpg densenet121_epoch5.pth --gpu
```

The following will return the top 5 most likely classes using a vgg13 checkpoint in the checkpoints directory executing on the GPU and map classes to real names using a mapping file:

```
python predict.py flowers/test/28/image_05230.jpg checkpoints/vgg13_bn_epoch5.pth --gpu --top_k 5 --category_names cat_to_name.json
```
## Unit Testing
## Training
Run unit tests for training using the GPU (required to generate checkpoints for inference GPU test cases):

```
python -m unittest test_model_helper.TrainingGpuTestCase
```

Run unit tests for training using the CPU (required to generate checkpoints for inference CPU test cases):

```
python -m unittest test_model_helper.TrainingCpuTestCase
```
## Inference
Run unit tests for inference using the GPU (assumes checkpoints have been generated):

```
python -m unittest test_model_helper.InferenceGpuTestCase
```

Run unit tests for inference using the CPU (assumes checkpoints have been generated):

```
python -m unittest test_model_helper.InferenceCpuTestCase
```
