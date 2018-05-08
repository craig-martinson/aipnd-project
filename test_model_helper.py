import unittest
import torch
from torchvision import datasets, transforms
import utility
import model_helper
import os
import json

data_dir = 'flowers'
testing_dir = 'testing'
gpu_epochs = 7
cpu_epochs = 7
category_names = 'cat_to_name.json'
hidden_units = 512
learning_rate = 0.001
test_image = 'flowers/test/28/image_05230.jpg'
correct_prediction_class = '28'
correct_prediction_category = 'stemless gentian'
num_workers = 4
num_cpu_threads = 16
top_k = 5


def train_test(tester, arch, enable_gpu):
    pin_memory = True if enable_gpu else False
    dataloaders, class_to_idx = model_helper.get_dataloders(data_dir,
                                                            enable_gpu,
                                                            num_workers,
                                                            pin_memory)

    model, optimizer, criterion = model_helper.create_model(arch,
                                                            hidden_units,
                                                            learning_rate,
                                                            class_to_idx)
    if enable_gpu:
        model.cuda()
    else:
        torch.set_num_threads(num_cpu_threads)

    epochs = gpu_epochs if enable_gpu else cpu_epochs

    model_helper.train(model,
                       criterion,
                       optimizer,
                       epochs,
                       dataloaders['training'],
                       dataloaders['validation'],
                       enable_gpu)

    checkpoint_dir = testing_dir + '/gpu' if enable_gpu else '/cpu'

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint = checkpoint_dir + '/' + arch + '_checkpoint.pth'

    model_helper.save_checkpoint(checkpoint,
                                 model,
                                 optimizer,
                                 arch,
                                 hidden_units,
                                 epochs)


def predict_test(tester, arch, enable_gpu):
    checkpoint_dir = testing_dir + '/gpu' if enable_gpu else '/cpu'
    checkpoint = checkpoint_dir + '/' + arch + '_checkpoint.pth'

    model = model_helper.load_checkpoint(checkpoint, True)

    if enable_gpu:
        model.cuda()

    probs, classes = model_helper.predict(
        test_image, model, enable_gpu, top_k)

    tester.assertEqual(len(classes), top_k, 'Incorrect number of results')
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
