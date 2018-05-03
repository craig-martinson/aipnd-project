# Image Classifier Project

Project code for Udacity's AI Programming with Python Nanodegree program.

## Training Examples
The following will train a densenet121 model executing on the GPU to a validation accuracy of 92.4%:

```python train.py flowers --arch densenet121 --gpu --epochs 4```

The following will train a vgg16 model executing on the GPU to a validation accuracy of 87.3%:

```python train.py flowers --arch vgg16 --gpu --epochs 8```

## Prediction Examples
The following will return the top 5 most likely classes using a pre-trained densenet121 model executing on the GPU:

```python predict.py flowers/test/28/image_05230.jpg checkpoints/densenet121_epoch1.pth --gpu --top_k 5```

The following will return the top 5 most likely classes using a pre-trained densenet121 model executing on the GPU and map categories to real names using a mapping file:

```python predict.py flowers/test/28/image_05230.jpg checkpoints/densenet121_epoch1.pth --gpu --top_k 5 --category_names cat_to_name.json```
