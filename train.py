import argparse
from time import time
import torch
from torchvision import datasets, transforms
import utility
import model_helper
import os


def get_input_args():
    parser = argparse.ArgumentParser()

    # Add positional arguments
    parser.add_argument('data_dir', type=str,
                        help='Set directory to training images')

    # Add optional arguments
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Set directory to save checkpoints')

    parser.add_argument('--arch', dest='arch', default='vgg16', action='store',
                        choices=['vgg16', 'densenet121'], help='Model architecture to use for training')

    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Set learning rate hyperparameter')

    parser.add_argument('--hidden_units', type=int, default=512,
                        help='Set number of hidden units hyperparameter')

    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of epochs used to train model')

    parser.add_argument('--gpu', dest='gpu',
                        action='store_true', help='Use GPU for training')
    parser.set_defaults(gpu=False)

    return parser.parse_args()


def main():
    start_time = time()

    in_args = get_input_args()

    # Check for GPU
    use_gpu = torch.cuda.is_available() and in_args.gpu

    # Print parameter information
    print("Training on {} using {}".format(
        "GPU" if use_gpu else "CPU", in_args.arch))

    print("Learning rate:{}, Hidden Units:{}, Epochs:{}".format(
        in_args.learning_rate, in_args.hidden_units, in_args.epochs))

    # Create save directory if required
    if not os.path.exists(in_args.save_dir):
        os.makedirs(in_args.save_dir)

    # Set data paths
    train_dir = in_args.data_dir + '/train'
    valid_dir = in_args.data_dir + '/valid'
    test_dir = in_args.data_dir + '/test'

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
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_gpu else {}
    dataloaders = {
        'training': torch.utils.data.DataLoader(image_datasets['training'], batch_size=64, shuffle=True, **kwargs),
        'validation': torch.utils.data.DataLoader(image_datasets['validation'], batch_size=64, shuffle=True, **kwargs),
        'testing': torch.utils.data.DataLoader(image_datasets['testing'], batch_size=64, shuffle=True, **kwargs)
    }

    # Create model
    model, optimizer, criterion = model_helper.create_model(in_args.arch,
                                                            in_args.hidden_units,
                                                            in_args.learning_rate,
                                                            image_datasets['training'].class_to_idx)

    # Move tensors to GPU if available
    if use_gpu:
        model.cuda()
        criterion.cuda()

    # Train the network using traning data
    model_helper.train(model,
                       criterion,
                       optimizer,
                       in_args.epochs,
                       dataloaders['training'],
                       dataloaders['validation'],
                       use_gpu)

    # Save trained model
    file_path = in_args.save_dir + '/' + in_args.arch + \
        '_epoch' + str(in_args.epochs) + '.pth'

    model_helper.save_checkpoint(file_path,
                                 model,
                                 optimizer,
                                 in_args.arch,
                                 in_args.hidden_units,
                                 in_args.epochs)

    # Do validation on the test set
    test_loss, accuracy = model_helper.validate(
        model, criterion, dataloaders['testing'], use_gpu)
    print("Post load Validation Accuracy: {:.3f}".format(accuracy))

    # Prediction
    image_path = 'flowers/test/28/image_05230.jpg'
    print("Predication for: {}".format(image_path))
    probs, classes = model_helper.predict(image_path, model, use_gpu)
    print(probs)
    print(classes)

    # Computes overall runtime in seconds & prints it in hh:mm:ss format
    end_time = time()
    utility.print_elapsed_time(end_time - start_time)


# Call to main function to run the program
if __name__ == "__main__":
    main()
