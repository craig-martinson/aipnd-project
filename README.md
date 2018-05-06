# PyTorch Transfer Learning

PyTorch transfer learning example developed as part of Udacity's AI Programming with Python Nanodegree program.

## Getting Started
### Environment
Tested on the following:
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
python train.py [-h] [--save_dir SAVE_DIR] [--arch {vgg,densenet,resnet}]
                [--learning_rate LEARNING_RATE] [--hidden_units HIDDEN_UNITS]
                [--epochs EPOCHS] [--gpu] [--num_workers NUM_WORKERS]
                [--pin_memory]
                data_dir
```
### Examples
The following will train a densenet model executing on the GPU for 5 epochs:

```
python train.py flowers --arch densenet --epochs 5 --gpu --pin_memory --num_workers 4
```

The following will train a vgg model executing on the GPU for 5 epochs:

```
python train.py flowers --arch vgg --epochs 5 --gpu --pin_memory --num_workers 4
```

The following will train a densenet model executing on the CPU for 3 epochs:

```
python train.py flowers --arch densenet --epochs 3
```

## Inference
### Usage
```
python predict.py [-h] [--top_k TOP_K] [--category_names CATEGORY_NAMES]
                  [--gpu] [--verbose]
                  input checkpoint
```
### Examples
The following will return the most likely class using a pre-trained densenet model executing on the GPU:

```
python predict.py flowers/test/28/image_05230.jpg checkpoints/densenet_epoch5.pth --gpu
```

The following will return the top 5 most likely classes using a pre-trained densenet model executing on the GPU and map classes to real names using a mapping file:

```
python predict.py flowers/test/28/image_05230.jpg checkpoints/densenet_epoch5.pth --gpu --top_k 5 --category_names cat_to_name.json
```
## Unit Testing
## Training
Run unit tests for training using the GPU (require to generate checkpoints for prediction test cases):

```
python -m unittest test_model_helper.TrainingGpuTestCase
```

Run unit tests for training using the CPU (takes a long time):

```
python -m unittest test_model_helper.TrainingCpuTestCase
```
## Inference
Run unit tests for prediction using the GPU (assumes checkpoints have been generated):

```
python -m unittest test_model_helper.PredictionGpuTestCase
```

Run unit tests for prediction using the CPU (assumes checkpoints have been generated):

```
python -m unittest test_model_helper.PredictionCpuTestCase
```


