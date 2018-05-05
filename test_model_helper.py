import unittest
import torch
from torchvision import datasets, transforms
import utility
import model_helper
import os
import json

testing_dir = 'testing'
densenet_checkpoint = testing_dir + '/densenet_checkpoint.pth'
vgg_checkpoint = testing_dir + '/vgg_checkpoint.pth'
resnet_checkpoint = testing_dir + '/resnet_checkpoint.pth'


class TrainingGpuTestCase(unittest.TestCase):
    train_dir = 'flowers/train'
    valid_dir = 'flowers/valid'
    test_dir = 'flowers/test'
    category_names = 'cat_to_name.json'
    hidden_units = 512
    learning_rate = 0.001
    gpu_epochs = 5

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

    def test_densenet(self):
        kwargs = {'num_workers': 1, 'pin_memory': True}
        training_dataloder = torch.utils.data.DataLoader(
            self.image_datasets['training'], batch_size=64, shuffle=True, **kwargs)
        validation_dataloder = torch.utils.data.DataLoader(
            self.image_datasets['validation'], batch_size=64, shuffle=True, **kwargs)

        model, optimizer, criterion = model_helper.create_model('densenet',
                                                                self.hidden_units,
                                                                self.learning_rate,
                                                                self.image_datasets['training'].class_to_idx)
        model.cuda()

        model_helper.train(model,
                           criterion,
                           optimizer,
                           self.gpu_epochs,
                           training_dataloder,
                           validation_dataloder,
                           True)

        if not os.path.exists(testing_dir):
            os.makedirs(testing_dir)

        model_helper.save_checkpoint(densenet_checkpoint,
                                     model,
                                     optimizer,
                                     'densenet',
                                     self.hidden_units,
                                     self.gpu_epochs)

    def test_vgg(self):
        kwargs = {'num_workers': 1, 'pin_memory': True}
        training_dataloder = torch.utils.data.DataLoader(
            self.image_datasets['training'], batch_size=64, shuffle=True, **kwargs)
        validation_dataloder = torch.utils.data.DataLoader(
            self.image_datasets['validation'], batch_size=64, shuffle=True, **kwargs)

        model, optimizer, criterion = model_helper.create_model('vgg',
                                                                self.hidden_units,
                                                                self.learning_rate,
                                                                self.image_datasets['training'].class_to_idx)
        model.cuda()

        model_helper.train(model,
                           criterion,
                           optimizer,
                           self.gpu_epochs,
                           training_dataloder,
                           validation_dataloder,
                           True)

        if not os.path.exists(testing_dir):
            os.makedirs(testing_dir)

        model_helper.save_checkpoint(vgg_checkpoint,
                                     model,
                                     optimizer,
                                     'vgg',
                                     self.hidden_units,
                                     self.gpu_epochs)

    def test_resnet(self):
        kwargs = {'num_workers': 1, 'pin_memory': True}
        training_dataloder = torch.utils.data.DataLoader(
            self.image_datasets['training'], batch_size=64, shuffle=True, **kwargs)
        validation_dataloder = torch.utils.data.DataLoader(
            self.image_datasets['validation'], batch_size=64, shuffle=True, **kwargs)

        model, optimizer, criterion = model_helper.create_model('resnet',
                                                                self.hidden_units,
                                                                self.learning_rate,
                                                                self.image_datasets['training'].class_to_idx)
        model.cuda()

        model_helper.train(model,
                           criterion,
                           optimizer,
                           self.gpu_epochs,
                           training_dataloder,
                           validation_dataloder,
                           True)

        if not os.path.exists(testing_dir):
            os.makedirs(testing_dir)

        model_helper.save_checkpoint(resnet_checkpoint,
                                     model,
                                     optimizer,
                                     'resnet',
                                     self.hidden_units,
                                     self.gpu_epochs)


class TrainingCpuTestCase(unittest.TestCase):
    train_dir = 'flowers/train'
    valid_dir = 'flowers/valid'
    test_dir = 'flowers/test'
    category_names = 'cat_to_name.json'
    hidden_units = 512
    learning_rate = 0.001
    cpu_epochs = 1

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

    def test_densenet(self):
        kwargs = {}
        training_dataloder = torch.utils.data.DataLoader(
            self.image_datasets['training'], batch_size=64, shuffle=True, **kwargs)
        validation_dataloder = torch.utils.data.DataLoader(
            self.image_datasets['validation'], batch_size=64, shuffle=True, **kwargs)

        model, optimizer, criterion = model_helper.create_model('densenet',
                                                                self.hidden_units,
                                                                self.learning_rate,
                                                                self.image_datasets['training'].class_to_idx)
        model_helper.train(model,
                           criterion,
                           optimizer,
                           self.cpu_epochs,
                           training_dataloder,
                           validation_dataloder,
                           False)

        if not os.path.exists(testing_dir):
            os.makedirs(testing_dir)

        densenet_cpu_checkpoint = testing_dir + '/densenet_cpu_checkpoint.pth'

        model_helper.save_checkpoint(densenet_cpu_checkpoint,
                                     model,
                                     optimizer,
                                     'densenet',
                                     self.hidden_units,
                                     self.cpu_epochs)


class PredictionGpuTestCase(unittest.TestCase):
    test_image = 'flowers/test/28/image_05230.jpg'
    category_names = 'cat_to_name.json'
    correct_prediction_class = '28'
    correct_prediction_category = 'stemless gentian'

    def test_densenet(self):
        model = model_helper.load_checkpoint(densenet_checkpoint, True)
        model.cuda()

        probs, classes = model_helper.predict(
            self.test_image, model, True, 5)

        self.assertEqual(len(classes), 5, 'Incorrect number of results')
        self.assertEqual(
            classes[0], self.correct_prediction_class, 'Incorrect prediction')

    def test_vgg(self):
        model = model_helper.load_checkpoint(vgg_checkpoint, True)
        model.cuda()

        probs, classes = model_helper.predict(
            self.test_image, model, True, 5)

        self.assertEqual(len(classes), 5, 'Incorrect number of results')
        self.assertEqual(
            classes[0], self.correct_prediction_class, 'Incorrect prediction')

    def test_resnet(self):
        model = model_helper.load_checkpoint(resnet_checkpoint, True)
        model.cuda()

        probs, classes = model_helper.predict(
            self.test_image, model, True, 5)

        self.assertEqual(len(classes), 5, 'Incorrect number of results')
        self.assertEqual(
            classes[0], self.correct_prediction_class, 'Incorrect prediction')

    def test_category(self):
        with open(self.category_names, 'r') as f:
            cat_to_name = json.load(f)

        model = model_helper.load_checkpoint(densenet_checkpoint, True)
        model.cuda()

        probs, classes = model_helper.predict(
            self.test_image, model, True)

        self.assertEqual(
            cat_to_name[classes[0]], self.correct_prediction_category, 'Incorrect prediction')


class PredictionCpuTestCase(unittest.TestCase):
    test_image = 'flowers/test/28/image_05230.jpg'
    category_names = 'cat_to_name.json'
    correct_prediction_class = '28'
    correct_prediction_category = 'stemless gentian'

    def test_densenet(self):
        model = model_helper.load_checkpoint(densenet_checkpoint, True)

        probs, classes = model_helper.predict(
            self.test_image, model, False, 5)

        self.assertEqual(len(classes), 5, 'Incorrect number of results')
        self.assertEqual(
            classes[0], self.correct_prediction_class, 'Incorrect prediction')

    def test_vgg(self):
        model = model_helper.load_checkpoint(vgg_checkpoint, True)

        probs, classes = model_helper.predict(
            self.test_image, model, False, 5)

        self.assertEqual(len(classes), 5, 'Incorrect number of results')
        self.assertEqual(
            classes[0], self.correct_prediction_class, 'Incorrect prediction')

    def test_resnet(self):
        model = model_helper.load_checkpoint(resnet_checkpoint, True)

        probs, classes = model_helper.predict(
            self.test_image, model, False, 5)

        self.assertEqual(len(classes), 5, 'Incorrect number of results')
        self.assertEqual(
            classes[0], self.correct_prediction_class, 'Incorrect prediction')

    def test_category(self):
        with open(self.category_names, 'r') as f:
            cat_to_name = json.load(f)

        model = model_helper.load_checkpoint(densenet_checkpoint, True)

        probs, classes = model_helper.predict(
            self.test_image, model, False)

        self.assertEqual(
            cat_to_name[classes[0]], self.correct_prediction_category, 'Incorrect prediction')


if __name__ == '__main__':
    unittest.main()
