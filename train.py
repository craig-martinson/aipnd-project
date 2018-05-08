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
                        help='Directory used to locate source images')

    # Add optional arguments
    parser.add_argument('--save_dir', type=str,
                        help='Directory used to save checkpoints')

    valid_archs = {'densenet121',
                   'densenet161',
                   'densenet201',
                   'vgg13_bn',
                   'vgg16_bn',
                   'vgg19_bn',
                   'resnet18',
                   'resnet34',
                   'resnet50'
                   }

    parser.add_argument('--arch', dest='arch', default='vgg', action='store',
                        choices=valid_archs,
                        help='Model architecture to use for training')

    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate hyperparameter')

    parser.add_argument('--hidden_units', type=int, default=512,
                        help='Number of hidden units hyperparameter')

    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs used to train model')

    parser.add_argument('--gpu', dest='gpu',
                        action='store_true', help='Use GPU for training')
    parser.set_defaults(gpu=False)

    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of subprocesses to use for data loading')

    parser.add_argument('--pin_memory', dest='pin_memory',
                        action='store_true', help='Request data loader to copy tensors into CUDA pinned memory')
    parser.set_defaults(pin_memory=False)

    return parser.parse_args()


def main():
    start_time = time()

    in_args = get_input_args()

    # Check for GPU
    use_gpu = torch.cuda.is_available() and in_args.gpu

    # Print parameter information
    if use_gpu:
        print("Training on {} using {} worker(s)".format(
            "GPU with pinned memory" if in_args.pin_memory else "GPU", in_args.num_workers))
    else:
        print("Training on CPU.")

    print("Architecture:{}, Learning rate:{}, Hidden Units:{}, Epochs:{}".format(
        in_args.arch, in_args.learning_rate, in_args.hidden_units, in_args.epochs))

    # Get dataloaders for training
    dataloaders, class_to_idx = model_helper.get_dataloders(in_args.data_dir,
                                                            use_gpu,
                                                            in_args.num_workers,
                                                            in_args.pin_memory)

    # Create model
    model, optimizer, criterion = model_helper.create_model(in_args.arch,
                                                            in_args.hidden_units,
                                                            in_args.learning_rate,
                                                            class_to_idx)

    # Move tensors to GPU if available
    if use_gpu:
        model.cuda()
        criterion.cuda()

    # Train the network
    model_helper.train(model,
                       criterion,
                       optimizer,
                       in_args.epochs,
                       dataloaders['training'],
                       dataloaders['validation'],
                       use_gpu)

    # Save trained model
    if in_args.save_dir:

        # Create save directory if required
        if not os.path.exists(in_args.save_dir):
            os.makedirs(in_args.save_dir)

         # Save checkpoint in save directory
        file_path = in_args.save_dir + '/' + in_args.arch + \
            '_epoch' + str(in_args.epochs) + '.pth'
    else:
        # Save checkpoint in current directory
        file_path = in_args.arch + \
            '_epoch' + str(in_args.epochs) + '.pth'

    model_helper.save_checkpoint(file_path,
                                 model,
                                 optimizer,
                                 in_args.arch,
                                 in_args.hidden_units,
                                 in_args.epochs)

    # Get prediction accuracy using test dataset
    test_loss, accuracy = model_helper.validate(
        model, criterion, dataloaders['testing'], use_gpu)
    print("Testing Accuracy: {:.3f}".format(accuracy))

    # Computes overall runtime in seconds & prints it in hh:mm:ss format
    end_time = time()
    utility.print_elapsed_time(end_time - start_time)


# Call to main function to run the program
if __name__ == "__main__":
    main()
