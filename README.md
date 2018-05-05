# PyTorch Transfer Learning

Project code for Udacity's AI Programming with Python Nanodegree program.

## Unit Testing
Run unit tests for training (require to generate checkpoints for prediction test case):

```python -m unittest test_model_helper.TrainingTestCase```

Run unit tests for prediction (assumes checkpoints have been generated):

```python -m unittest test_model_helper.PredictionTestCase```

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

