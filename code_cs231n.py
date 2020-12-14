from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy

RESOLUTIONS = [65, 97, 129, 224]
MODEL_NAMES = ["resnet18", "resnet34", "resnet50", "resnet101"]
DATA_DIRS = ["/lfs/1/katherinewu/birds200"]  #Change for each dataset

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss    
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            if(phase == 'train'):
                print('{:.4f}, {:.4f}, '.format(epoch_loss, epoch_acc), end='')
            else:
                print('{:.4f}, {:.4f}'.format(epoch_loss, epoch_acc))
           
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set.
    model_ft = None
    input_size = 0
    if model_name == "resnet18":
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = resolution
    elif model_name == "resnet34":
        model_ft = models.resnet34(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = resolution
    elif model_name == "resnet50":
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = resolution
    elif model_name == "resnet101":
        model_ft = models.resnet101(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = resolution
    return model_ft, input_size


def main():
    for resolution in RESOLUTIONS:
        num_dirs = 0
        for data_dir in DATA_DIRS:
            num_dirs += 1
            for model_name in MODEL_NAMES:
                
                print('resolution: ' + str(resolution) + ', data: ' + data_dir + ', model_name: ' + model_name)
                num_classes = 200 #Change for each dataset
                batch_size = 8
                num_epochs = 25
                feature_extract = False

                # Initialize the model for this run
                model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

                # Data augmentation and normalization for training
                # Just normalization for validation
                data_transforms = {
                    'train': transforms.Compose([
                        transforms.Resize((resolution, resolution)),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                    ]),
                    'val': transforms.Compose([
                        transforms.Resize((resolution, resolution)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                    ]),
                    'test': transforms.Compose([
                        transforms.Resize((resolution, resolution)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                    ]),
                }


                # Create training and validation datasets
                image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val', 'test']}
                # Create training and validation dataloaders
                dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val', 'test']}
                # Detect if we have a GPU available
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

                model_ft = model_ft.to(device)
                
                params_to_update = model_ft.parameters()
                params_to_update = []
                for name,param in model_ft.named_parameters():
                    if param.requires_grad == True:
                        params_to_update.append(param)
                            

                optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
                criterion = nn.CrossEntropyLoss()

                # Train and evaluate
                model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"))

                correct = 0
                total = 0
                with torch.no_grad():
                    for data in dataloaders_dict['test']:
                        images, labels = data
                        images = images.to(device)
                        labels = labels.to(device)
                        outputs = model_ft(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                print('Accuracy of the network on the test images: %d %%' % ( 100 * correct / total))                                                                                                                                                                                                                                                                                                                                                                                                                 


if __name__ == '__main__':
    main()