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
from torch.utils.data import Dataset

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy
import pandas as pd
from PIL import Image
from tqdm import tqdm


class CustomDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        path = self.dataframe.loc[index, 'path']
        label = self.dataframe.loc[index, 'label']
        image = Image.open(path).convert('RGB')

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
            ])
        image = transform(image)
        return image, label

class Classification:
    def __init__(self, data_dir, batch_size, num_epochs, learning_rate, device):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.device = device
        self.class_names = None

        self.dataloaders, self.dataset_sizes = self.load_data()
        print("Dataset loaded!")
        self.model = self.initialize_model()
        print("Model initialized!")

    def initialize_model(self):
        model = models.vgg19(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
            
        model.classifier = nn.Sequential(
                            nn.Linear(25088, 1024),  # Adjust the input size to match VGG19's output size
                            nn.ReLU(),
                            nn.Dropout(p=0.5),
                            nn.Linear(1024, len(self.class_names)),
                            nn.LogSoftmax(dim=1)
                        )
        model = model.to(self.device)

        return model
    

    def load_data(self):
        transform = transforms.Compose([
                    transforms.RandomRotation(10),      # rotate +/- 10 degrees
                    transforms.RandomHorizontalFlip(),  # reverse 50% of images
                    transforms.Resize(224),             # resize shortest side to 224 pixels
                    transforms.CenterCrop(224),         # crop longest side to 224 pixels at center
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406],
                                        [0.229, 0.224, 0.225])
                    ])

        paths=[]
        labels=[]
        for dirname, _, filenames in os.walk(os.path.join(self.data_dir,'train')):
            for filename in filenames:
                if filename[-4:]=='JPEG':
                    paths+=[(os.path.join(dirname, filename))]
                    label=dirname.split('/')[-1]
                    labels+=[label]

        val_paths=[]
        val_labels=[]
        for dirname, _, filenames in os.walk(os.path.join(self.data_dir,'val')):
            for filename in filenames:
                if filename[-4:]=='JPEG':
                    val_paths+=[(os.path.join(dirname, filename))]
                    label=dirname.split('/')[-1]
                    val_labels+=[label]

        self.class_names = sorted(set(labels))
        N = list(range(len(self.class_names)))
        normal_mapping = dict(zip(self.class_names, N))
        reverse_mapping = dict(zip(N, self.class_names))

        df=pd.DataFrame(columns=['path','label'])
        df['path']=paths
        df['label']=labels
        df['label']=df['label'].map(normal_mapping)

        tdf=pd.DataFrame(columns=['path','label'])
        tdf['path']=val_paths
        tdf['label']=val_labels
        tdf['label']=tdf['label'].map(normal_mapping)
        

        train_ds = CustomDataset(df)
        val_ds = CustomDataset(tdf)

        image_datasets = {'train': train_ds, 'val': val_ds}
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=self.batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
        return dataloaders, dataset_sizes
    

    def train_model(self):
        print("Training started!")
        since = time.time()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        for epoch in range(self.num_epochs):
            print('Epoch {}/{}'.format(epoch, self.num_epochs - 1))
            print('-' * 10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Wrap the dataloader with tqdm for a progress bar
                progress_bar = tqdm(self.dataloaders[phase], desc=f"Phase: {phase}", leave=False)
                for inputs, labels in progress_bar:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                    # Optionally, update the progress bar with current loss
                    progress_bar.set_postfix(loss=loss.item())

                if phase == 'train':
                    scheduler.step()

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




