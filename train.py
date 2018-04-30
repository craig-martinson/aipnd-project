import numpy as np
import argparse
from time import time
import json
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from collections import OrderedDict


def validate_model(model, criterion, data_loader, use_gpu):
    # Put model in inference mode
    model.eval()

    accuracy = 0
    test_loss = 0
    for inputs, labels in iter(data_loader):

        # Set volatile to True so we don't save the history
        if use_gpu:
            inputs = Variable(inputs.float().cuda(), volatile=True)
            labels = Variable(labels.long().cuda(), volatile=True)
        else:
            inputs = Variable(inputs, volatile=True)
            labels = Variable(labels, volatile=True)

        output = model.forward(inputs)
        test_loss += criterion(output, labels).data[0]

        # Model's output is log-softmax,
        # take exponential to get the probabilities
        ps = torch.exp(output).data

        # Model's output is softmax
        # ps = output.data

        # Class with highest probability is our predicted class,
        equality = (labels.data == ps.max(1)[1])

        # Accuracy is number of correct predictions divided by all predictions, just take the mean
        accuracy += equality.type_as(torch.FloatTensor()).mean()

    return test_loss/len(data_loader), accuracy/len(data_loader)


def train_model(model, criterion, optimizer, epochs, training_data_loader, validation_data_loader, use_gpu):
    # Ensure model in training mode
    model.train()

    # Train the network using training data
    print_every = 40
    steps = 0

    for epoch in range(epochs):
        running_loss = 0

        # Get inputs and labels from training set
        for inputs, labels in iter(training_data_loader):
            steps += 1

            # Move tensors to GPU if available
            if use_gpu:
                inputs = Variable(inputs.float().cuda())
                labels = Variable(labels.long().cuda())
            else:
                inputs = Variable(inputs)
                labels = Variable(labels)

            # Set gradients to zero
            optimizer.zero_grad()

            # Forward pass to calculate logits
            output = model.forward(inputs)

            # Calculate loss (how far is prediction from label)
            loss = criterion(output, labels)

            # Backward pass to calculate gradients
            loss.backward()

            # Update weights using optimizer (add gradients to weights)
            optimizer.step()

            # Track the loss as we are training the network
            running_loss += loss.data[0]

            if steps % print_every == 0:
                test_loss, accuracy = validate_model(model,
                                                     criterion,
                                                     validation_data_loader,
                                                     use_gpu)

                print("Epoch: {}/{} ".format(epoch+1, epochs),
                      "Training Loss: {:.3f} ".format(
                          running_loss/print_every),
                      "Test Loss: {:.3f} ".format(test_loss),
                      "Test Accuracy: {:.3f}".format(accuracy))

                running_loss = 0

                # Put model back in training mode
                model.train()


def create_model(arch, class_to_idx):
    # Load pretrained DenseNet model
    model = models.densenet121(pretrained=True)
    #model = models.vgg16(pretrained=True)

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    # Replace classifier, ensure output sizes matches number of classes
    # input_size = 224 * 224 * 3
    output_size = 102

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(1024, 500)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(500, output_size)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier

    # Set training parameters
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    # optimizer = optim.SGD(parameters, lr=0.001)
    optimizer = optim.Adam(parameters, lr=0.001)
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.NLLLoss()

    # Swap keys and items
    model.class_to_idx = {class_to_idx[k]: k for k in class_to_idx}

    return model, optimizer, criterion


def save_checkpoint(file_path, model, optimizer, total_epochs):
    # Save the checkpoint
    state = {
        'epoch': total_epochs,
        'arch': 'densenet121',
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'class_to_idx': model.class_to_idx
    }

    torch.save(state, file_path)


def get_input_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dir', type=str, default='flowers',
                        help='Path to the image files')

    parser.add_argument('--arch', type=str, default='vgg',
                        help='CNN model architecture to use for image classification')

    parser.add_argument('--epochs', type=int, default='3',
                        help='Number of epochs used to train model')

    return parser.parse_args()


def print_elapsed_time(total_time):
    hh = int(total_time / 3600)
    mm = int((total_time % 3600) / 60)
    ss = int((total_time % 3600) % 60)
    print(
        "\n** Total Elapsed Runtime: {:0>2}:{:0>2}:{:0>2}".format(hh, mm, ss))


def main():
    start_time = time()

    in_args = get_input_args()

    # Check for GPU
    use_gpu = torch.cuda.is_available()
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_gpu else {}
    print("Trainig with {}".format("GPU" if use_gpu else "CPU"))

    # Set data paths
    data_dir = in_args.dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Define transforms for the training, validation, and testing sets
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

    # Load the datasets with ImageFolder
    image_datasets = {
        'training': datasets.ImageFolder(train_dir, transform=data_transforms['training']),
        'validation': datasets.ImageFolder(valid_dir, transform=data_transforms['validation']),
        'testing': datasets.ImageFolder(test_dir, transform=data_transforms['testing'])
    }

    # Using the image datasets and the transforms, define the dataloaders
    dataloaders = {
        'training': torch.utils.data.DataLoader(image_datasets['training'], batch_size=64, shuffle=True, **kwargs),
        'validation': torch.utils.data.DataLoader(image_datasets['validation'], batch_size=64, shuffle=True, **kwargs),
        'testing': torch.utils.data.DataLoader(image_datasets['testing'], batch_size=64, shuffle=True, **kwargs)
    }

    # Load category mapping dictionary
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    # Load pretrained model
    model, optimizer, criterion = create_model(
        in_args.arch, image_datasets['training'].class_to_idx)

    # Move tensors to GPU if available
    if use_gpu:
        model.cuda()
        criterion.cuda()

    # Train the network using traning data
    train_model(model,
                criterion,
                optimizer,
                in_args.epochs,
                dataloaders['training'],
                dataloaders['validation'],
                use_gpu)

    # Save trained model
    save_checkpoint('test.pth', model, optimizer, in_args.epochs)

    # Do validation on the test set
    test_loss, accuracy = validate_model(
        model, criterion, dataloaders['testing'], use_gpu)
    print("Validation Accuracy: {:.3f}".format(accuracy))

    # Computes overall runtime in seconds & prints it in hh:mm:ss format
    end_time = time()
    print_elapsed_time(end_time - start_time)


# Call to main function to run the program
if __name__ == "__main__":
    main()
