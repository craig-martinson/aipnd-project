import numpy as np
import matplotlib.pyplot as plt
import time
import json
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from collections import OrderedDict

# Monitor GPU with: watch --color -n1.0 gpustat --color
# Get nvidia driver vresion: nvidia-smi



def validate_model(model, criterion, data_loader):
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

def train_model(model, criterion, optimizer, epochs, training_data_loader, validation_data_loader):
    # Ensure model in training mode
    model.train()

    # Train the network using training data
    print_every = 40
    steps = 0

    for e in range(epochs):
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
                test_loss, accuracy = validate_model(model, criterion, validation_data_loader)

                print("Epoch: {}/{} ".format(e+1, epochs),
                        "Training Loss: {:.3f} ".format(running_loss/print_every),
                        "Test Loss: {:.3f} ".format(test_loss),
                        "Test Accuracy: {:.3f}".format(accuracy))

                running_loss = 0

                # Put model back in training mode
                model.train()
                
def get_class_name(label):
    #model.class_to_idx
    c = list(model.class_to_idx.keys())[list(model.class_to_idx.values()).index(label)]

    # return cat_to_name[str(label + 1)]
    return cat_to_name[str(c)]

def display_results(model, data_loader):
    columns = 4
    rows = 4
    max_images = columns * rows
    image_count = 0
    fig = plt.figure(figsize=(24, 24))

    for inputs, labels in iter(data_loader):

        if use_gpu:
            var_inputs = Variable(inputs.float().cuda(), volatile=True)
        else:       
            var_inputs = Variable(inputs, volatile=True)

        output = model.forward(var_inputs)
        ps = torch.exp(output).data

        for batch_index in range(0, len(labels)):    
            if image_count < max_images: 
                image = inputs[batch_index].numpy().transpose((1, 2, 0))
                
                # Remove normalization
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                image = std * image + mean
                image = np.clip(image, 0, 1)

                # Add the sub image to the grid
                fig.add_subplot(rows, columns, image_count + 1)

                # Get index of highest probability in batch
                prob, prob_index = torch.max(ps[batch_index], 0)

                # Move to CPU if needed
                highest_prob = prob.cpu().numpy()[0] if use_gpu else prob.numpy()[0]
                highest_prob_index = prob_index.cpu().numpy()[0] if use_gpu else prob_index.numpy()[0]

                # Lookup name in category dictionary
                pred_title = get_class_name(highest_prob_index)

                # Get image title from label
                actual_title = get_class_name(labels[batch_index])

                plt.title("{} : {} ({:.3f})".format(actual_title, pred_title, highest_prob))
                plt.axis('off')
                plt.imshow(image)
                image_count += 1

        if image_count >= max_images:
            break

    plt.show()

def save_checkpoint(model, filepath):
    checkpoint = {'state_dict': model.state_dict()}

    torch.save(checkpoint, filepath)

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)

    model.load_state_dict(checkpoint['state_dict'])

    return model

# Check for GPU
use_gpu = torch.cuda.is_available()
print("GPU {}".format("Enabled" if use_gpu else "Disabled"))
kwargs = {'num_workers': 1, 'pin_memory': True} if use_gpu else {}

# Load category mapping dictionary
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# Set data paths
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# Define transforms for the training, validation, and testing sets
data_transforms = {
    'training' : transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])]),
                                                            
    'validation' : transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])]),

    'testing' : transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
}

# Load the datasets with ImageFolder
image_datasets = {
    'training' : datasets.ImageFolder(train_dir, transform=data_transforms['training']),
    'validation' : datasets.ImageFolder(valid_dir, transform=data_transforms['validation']),
    'testing' : datasets.ImageFolder(test_dir, transform=data_transforms['testing'])
}

# Using the image datasets and the transforms, define the dataloaders
dataloaders = {
    'training' : torch.utils.data.DataLoader(image_datasets['training'], batch_size=64, shuffle=True, **kwargs),
    'validation' : torch.utils.data.DataLoader(image_datasets['validation'], batch_size=64, shuffle=True, **kwargs),
    'testing' : torch.utils.data.DataLoader(image_datasets['testing'], batch_size=64, shuffle=True, **kwargs)
}

enable_training = False

# Load pretrained DenseNet model
model = models.densenet121(pretrained=True)
#model = models.vgg16(pretrained=True)

# Get class to index mapping
model.class_to_idx = image_datasets['training'].class_to_idx

# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

# Replace classifier, ensure output sizes matches number of classes
input_size = 224 * 224 * 3
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
epochs = 10

# Move tensors to GPU if available
if use_gpu:
    model.cuda()
    criterion.cuda()

if enable_training:
    # Train new classifier
    train_model(model, criterion, optimizer, epochs, dataloaders['training'], dataloaders['validation'])
    save_checkpoint(model, 'checkpoint.pth')

if not enable_training:
    model = load_checkpoint('checkpoint.pth')

# Validate trained model using images the model has never seen
test_loss, accuracy = validate_model(model, criterion, dataloaders['testing'])
print("Validation Accuracy: {:.3f}".format(accuracy))

# Display a subset of results
display_results(model, dataloaders['testing'])
