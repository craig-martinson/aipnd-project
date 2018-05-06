import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from collections import OrderedDict
import utility
from PIL import Image


def get_model_from_arch(arch, hidden_units):
    ''' Load an existing PyTorch model, freeze parameters and subsitute classifier.
    '''
    if arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        classifier_input_size = model.classifier.in_features
    elif arch == 'densenet161':
        model = models.densenet161(pretrained=True)
        classifier_input_size = model.classifier.in_features
    elif arch == 'densenet201':
        model = models.densenet201(pretrained=True)
        classifier_input_size = model.classifier.in_features
    elif arch == 'vgg13':
        model = models.vgg13(pretrained=True)
        classifier_input_size = model.classifier[0].in_features
    elif arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        classifier_input_size = model.classifier[0].in_features
    elif arch == 'vgg19':
        model = models.vgg19(pretrained=True)
        classifier_input_size = model.classifier[0].in_features
    elif arch == 'resnet18':
        model = models.resnet18(pretrained=True)
        classifier_input_size = model.fc.in_features
    elif arch == 'resnet34':
        model = models.resnet34(pretrained=True)
        classifier_input_size = model.fc.in_features
    elif arch == 'resnet50':
        model = models.resnet50(pretrained=True)
        classifier_input_size = model.fc.in_features
    else:
        raise RuntimeError("Unknown model")

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    # Replace classifier, ensure input and output sizes match
    classifier_output_size = 102

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(classifier_input_size, hidden_units)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(hidden_units, classifier_output_size)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    if arch.startswith('densenet'):
        model.classifier = classifier
    elif arch.startswith('vgg'):
        model.classifier = classifier
    elif arch.startswith('resnet'):
        model.fc = classifier

    return model


def create_model(arch, hidden_units, learning_rate, class_to_idx):
    ''' Create a deep learning model from existing PyTorch model.
    '''
    # Load pre-trained model
    model = get_model_from_arch(arch, hidden_units)

    # Set training parameters
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(parameters, lr=learning_rate)
    criterion = nn.NLLLoss()

    # Swap keys and items
    model.class_to_idx = {class_to_idx[k]: k for k in class_to_idx}

    return model, optimizer, criterion


def save_checkpoint(file_path, model, optimizer, arch, hidden_units, epochs):
    ''' Save a trained deep learning model.
    '''
    state = {
        'arch': arch,
        'hidden_units': hidden_units,
        'epochs': epochs,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'class_to_idx': model.class_to_idx
    }

    torch.save(state, file_path)

    print("Checkpoint Saved: '{}'".format(file_path))


def load_checkpoint(file_path, verbose=False):
    ''' Load a previously trained deep learning model.
    '''
    state = torch.load(file_path)

    # Get pre-trained model
    model = get_model_from_arch(state['arch'], state['hidden_units'])

    # Load model state
    model.load_state_dict(state['state_dict'])
    model.class_to_idx = state['class_to_idx']

    if verbose:
        print("Checkpoint Loaded: '{}' (arch={}, hidden_units={}, epochs={})".format(
            file_path, state['arch'], state['hidden_units'], state['epochs']))

    return model


def validate(model, criterion, data_loader, use_gpu):
    ''' Validate a deep learning model against a validation dataset.
    '''
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


def train(model,
          criterion,
          optimizer,
          epochs,
          training_data_loader,
          validation_data_loader,
          use_gpu):
    ''' Train a deep learning model using a training dataset.
    '''
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
                validation_loss, validation_accuracy = validate(model,
                                                                criterion,
                                                                validation_data_loader,
                                                                use_gpu)

                print("Epoch: {}/{} ".format(epoch+1, epochs),
                      "Training Loss: {:.3f} ".format(
                          running_loss/print_every),
                      "Validation Loss: {:.3f} ".format(validation_loss),
                      "Validation Accuracy: {:.3f}".format(validation_accuracy))

                running_loss = 0

                # Put model back in training mode
                model.train()


def predict(image_path, model, use_gpu, topk=5):
    ''' Predict the class (or classes) of an image using a previously trained deep learning model.
    '''
    # Put model in inference mode
    model.eval()

    image = Image.open(image_path)
    np_array = utility.process_image(image)
    tensor = torch.from_numpy(np_array)

    # Use GPU if available
    if use_gpu:
        var_inputs = Variable(tensor.float().cuda(), volatile=True)
    else:
        var_inputs = Variable(tensor, volatile=True).float()

    # Model is expecting 4d tensor, add another dimension
    var_inputs = var_inputs.unsqueeze(0)

    # Run image through model
    output = model.forward(var_inputs)

    # Model's output is log-softmax,
    # take exponential to get the probabilities
    ps = torch.exp(output).data.topk(topk)

    # Move results to CPU if needed
    probs = ps[0].cpu() if use_gpu else ps[0]
    classes = ps[1].cpu() if use_gpu else ps[1]

    # Map classes to indices
    mapped_classes = list()
    for label in classes.numpy()[0]:
        mapped_classes.append(model.class_to_idx[label])

    # Return results
    return probs.numpy()[0], mapped_classes
