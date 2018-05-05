# PyTorch Transfer Learning

Project code for Udacity's AI Programming with Python Nanodegree program.

## Sample data
You can download sample data using curl:

```curl -O https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz```

And extract using tar:
```mkdir flowers```
```tar -xvzf flower_data.tar.gz -C flowers```

## Unit Testing
Run unit tests for training using the GPU (require to generate checkpoints for prediction test cases):

```python -m unittest test_model_helper.TrainingGpuTestCase```

Run unit tests for training using the CPU:

```python -m unittest test_model_helper.TrainingCpuTestCase```

Run unit tests for prediction using the GPU (assumes checkpoints have been generated):

```python -m unittest test_model_helper.PredictionGpuTestCase```

Run unit tests for prediction using the CPU (assumes checkpoints have been generated):

```python -m unittest test_model_helper.PredictionCpuTestCase```

## Examples
### Training
The following will train a densenet model executing on the GPU:

```python train.py flowers --arch densenet --gpu --epochs 5```

The following will train a vgg model executing on the GPU:

```python train.py flowers --arch vgg --gpu --epochs 5```

The following will train a densenet model executing on the GPU:

```python train.py flowers --arch densenet --gpu --epochs 5```

### Prediction
The following will return the most likely class using a pre-trained densenet model executing on the GPU:

```python predict.py flowers/test/28/image_05230.jpg checkpoints/densenet_epoch5.pth --gpu```

The following will return the top 5 most likely classes using a pre-trained densenet model executing on the GPU and map classes to real names using a mapping file:

```python predict.py flowers/test/28/image_05230.jpg checkpoints/densenet_epoch5.pth --gpu --top_k 5 --category_names cat_to_name.json```

