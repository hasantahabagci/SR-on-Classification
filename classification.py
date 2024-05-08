"""
Authors: Hasan Taha Bagci (bagcih21@,tu.edu.tr), 
         Omer Faruk Aydin (aydinome21@itu.edu.tr)
"""



import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.utils.data

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy
import random
import argparse
from PIL import Image
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd



class Classification:
    def __init__(self, data_dir, batch_size, num_epochs, learning_rate, device):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.device = device

    def initialize_model(self):
        model = models.vgg19(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, 2)
        model = model.to(self.device)
        return model
    
    def load_data(self):
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        image_datasets = {x: torchvision.datasets.ImageFolder(os.path.join(self.data_dir, x), data_transforms[x]) for x in ['train', 'val']}
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=self.batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
        return dataloaders, dataset_sizes

    def initialize_optimizer(self):
        optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
        return optimizer
    
    def initialize_scheduler(self):
        scheduler = lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)
        return scheduler
    
    def train_model(self):
        since = time.time()
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0
        for epoch in range(self.num_epochs):
            print('Epoch {}/{}'.format(epoch, self.num_epochs - 1))
            print('-' * 10)
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()
                running_loss = 0.0
                running_corrects = 0
                for inputs, labels in self.dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    self.optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    self.scheduler.step()
                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = running_corrects.double() / self.dataset_sizes[phase]
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))
        self.model.load_state_dict(best_model_wts)
        return self.model
    
    def test_model(self):
        self.model.eval()
        running_corrects = 0
        for inputs, labels in self.dataloaders['val']:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
        acc = running_corrects.double() / self.dataset_sizes['val']
        print('Test Acc: {:.4f}'.format(acc))

    def visualize_model(self):
        self.model.eval()
        data_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        image_datasets = torchvision.datasets.ImageFolder(os.path.join(self.data_dir, 'val'), data_transforms)
        dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=self.batch_size, shuffle=True, num_workers=4)
        class_names = image_datasets.classes
        images, labels = next(iter(dataloaders))
        images = images.to(self.device)
        labels = labels.to(self.device)
        outputs = self.model(images)
        _, preds = torch.max(outputs, 1)
        fig, ax = plt.subplots(1, 4, figsize=(15, 15))
        for i in range(4):
            ax[i].imshow(images[i].cpu().numpy().transpose((1, 2, 0)))
            ax[i].set_title('Predicted: {}, Truth: {}'.format(class_names[preds[i]], class_names[labels[i]]))
            ax[i].axis('off')
        plt.show()

    def save_model(self):
        torch.save(self.model.state_dict(), 'model.pth')
        print('Model saved')


    def main(self):
        self.model = self.initialize_model()
        self.dataloaders, self.dataset_sizes = self.load_data()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self.initialize_optimizer()
        self.scheduler = self.initialize_scheduler()
        self.model = self.train_model()
        self.test_model()
        self.visualize_model()
        self.save_model()


