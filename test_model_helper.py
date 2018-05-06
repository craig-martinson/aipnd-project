import unittest
import torch
from torchvision import datasets, transforms
import utility
import model_helper
import os
import json

testing_dir = 'testing'
gpu_epochs = 5
cpu_epochs = 1
train_dir = 'flowers/train'
valid_dir = 'flowers/valid'
test_dir = 'flowers/test'
category_names = 'cat_to_name.json'
hidden_units = 512
learning_rate = 0.001
test_image = 'flowers/test/28/image_05230.jpg'
correct_prediction_class = '28'
correct_prediction_category = 'stemless gentian'

data_transforms = {
    'training': transforms.Compose([transforms.RandomRotation(30),
                                    transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])]),

    'validation': transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])]),

    'testing': transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406],
                                                        [0.229, 0.224, 0.225])])
}

image_datasets = {
    'training': datasets.ImageFolder(train_dir, transform=data_transforms['training']),
    'validation': datasets.ImageFolder(valid_dir, transform=data_transforms['validation']),
    'testing': datasets.ImageFolder(test_dir, transform=data_transforms['testing'])
}


def train_test(tester, arch, enable_gpu):
    kwargs = {'num_workers': 1, 'pin_memory': True} if enable_gpu else {}

    training_dataloder = torch.utils.data.DataLoader(
        image_datasets['training'], batch_size=64, shuffle=True, **kwargs)
    validation_dataloder = torch.utils.data.DataLoader(
        image_datasets['validation'], batch_size=64, shuffle=True, **kwargs)

    model, optimizer, criterion = model_helper.create_model(arch,
                                                            hidden_units,
                                                            learning_rate,
                                                            image_datasets['training'].class_to_idx)
    model.cuda()

    epochs = gpu_epochs if enable_gpu else cpu_epochs

    model_helper.train(model,
                       criterion,
                       optimizer,
                       epochs,
                       training_dataloder,
                       validation_dataloder,
                       enable_gpu)

    if not os.path.exists(testing_dir):
        os.makedirs(testing_dir)

    ptype = '_gpu_' if enable_gpu else '_cpu_'
    checkpoint = testing_dir + '/' + arch + ptype + 'checkpoint.pth'

    model_helper.save_checkpoint(checkpoint,
                                 model,
                                 optimizer,
                                 arch,
                                 hidden_units,
                                 epochs)


def predict_test(tester, arch, enable_gpu):
    ptype = '_gpu_' if enable_gpu else '_cpu_'
    checkpoint = testing_dir + '/' + arch + ptype + 'checkpoint.pth'

    model = model_helper.load_checkpoint(checkpoint, True)
    model.cuda()

    probs, classes = model_helper.predict(
        test_image, model, enable_gpu, 5)

    tester.assertEqual(len(classes), 5, 'Incorrect number of results')
    tester.assertEqual(
        classes[0], correct_prediction_class, 'Incorrect prediction')

    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)

    tester.assertEqual(
        cat_to_name[classes[0]], correct_prediction_category, 'Incorrect prediction')


class TrainingGpuTestCase(unittest.TestCase):

    def test_densenet121(self):
        train_test(self, 'densenet121', True)

    def test_densenet161(self):
        train_test(self, 'densenet161', True)

    def test_densenet201(self):
        train_test(self, 'densenet201', True)

    def test_vgg13_bn(self):
        train_test(self, 'vgg13_bn', True)

    def test_vgg16_bn(self):
        train_test(self, 'vgg16_bn', True)

    def test_vgg19_bn(self):
        train_test(self, 'vgg19_bn', True)

    def test_resnet18(self):
        train_test(self, 'resnet18', True)

    def test_resnet34(self):
        train_test(self, 'resnet34', True)

    def test_resnet50(self):
        train_test(self, 'resnet50', True)


class TrainingCpuTestCase(unittest.TestCase):

    def test_densenet121(self):
        train_test(self, 'densenet121', False)

    def test_densenet161(self):
        train_test(self, 'densenet161', False)

    def test_densenet201(self):
        train_test(self, 'densenet201', False)

    def test_vgg13_bn(self):
        train_test(self, 'vgg13_bn', False)

    def test_vgg16_bn(self):
        train_test(self, 'vgg16_bn', False)

    def test_vgg19_bn(self):
        train_test(self, 'vgg19_bn', False)

    def test_resnet18(self):
        train_test(self, 'resnet18', False)

    def test_resnet34(self):
        train_test(self, 'resnet34', False)

    def test_resnet50(self):
        train_test(self, 'resnet50', False)


class InferenceGpuTestCase(unittest.TestCase):

    def test_densenet121(self):
        predict_test(self, 'densenet121', True)

    def test_densenet161(self):
        predict_test(self, 'densenet161', True)

    def test_densenet201(self):
        predict_test(self, 'densenet201', True)

    def test_vgg13(self):
        predict_test(self, 'vgg13_bn', True)

    def test_vgg16(self):
        predict_test(self, 'vgg16_bn', True)

    def test_vgg19(self):
        predict_test(self, 'vgg19_bn', True)

    def test_resnet18(self):
        predict_test(self, 'resnet18', True)

    def test_resnet34(self):
        predict_test(self, 'resnet34', True)

    def test_resnet50(self):
        predict_test(self, 'resnet50', True)


class InferenceCpuTestCase(unittest.TestCase):

    def test_densenet121(self):
        predict_test(self, 'densenet121', False)

    def test_densenet161(self):
        predict_test(self, 'densenet161', False)

    def test_densenet201(self):
        predict_test(self, 'densenet201', False)

    def test_vgg13(self):
        predict_test(self, 'vgg13_bn', False)

    def test_vgg16(self):
        predict_test(self, 'vgg16_bn', False)

    def test_vgg19(self):
        predict_test(self, 'vgg19_bn', False)

    def test_resnet18(self):
        predict_test(self, 'resnet18', False)

    def test_resnet34(self):
        predict_test(self, 'resnet34', False)

    def test_resnet50(self):
        predict_test(self, 'resnet50', False)


if __name__ == '__main__':
    unittest.main()
