#%% Import Packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
from sklearn.metrics import accuracy_score

#%% transform and load data
transform = transforms.Compose([transforms.Resize(32),
                                transforms.Grayscale(num_output_channels=1),
                                transforms.ToTensor(),
                                transforms.Normalize(.5, .5)])

batch_size = 4
trainset = torchvision.datasets.ImageFolder(root= "train", transform=transform)
testset = torchvision.datasets.ImageFolder(root= "test", transform=transform)
train_data_loader = DataLoader(trainset, batch_size= batch_size, shuffle=True)
test_data_loader = DataLoader(testset, batch_size= batch_size, shuffle=True)

# %%Visualize the image
def imshow(img):
    img = img/2 + 0.5 #Un-normalize
    imgnp = img.numpy()
    plt.imshow(np.transpose(imgnp, (1,2,0)))
    plt.show()

# get some random images
imgiter = iter(train_data_loader)
images, label = next(imgiter)
imshow(torchvision.utils.make_grid(images, nrow=2))

# %%
class BinaryImageClassification(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,6,3) # 6, 30, 30
        self.pool = nn.MaxPool2d(2,2) #6, 15, 15
        self.conv2 = nn.Conv2d(6, 16, 3) # 16, 13, 13
        self.fc1 = nn.Linear(16*6*6, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x =self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

# %%training loop
num_epoch = 20
LR = 0.02
model = BinaryImageClassification()
loss_ftn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = LR)
for epoch in range(num_epoch):
    for i, data in enumerate(train_data_loader, 0):
        
        inputs, label = data

        # Zero grad
        optimizer.zero_grad

        #prediction
        output = model(inputs)

        #calculate loss
        loss = loss_ftn(output, label.reshape(-1, 1).float())

        #backward function
        loss.backward()

        #Update weight
        optimizer.step()
    
        if i%100 = 0:
            print(f'Epoch {epoch}/ {num_epoch}, step = {i+1}/ {len(train_data_loader)},'
            f'Loss: {loss.item():.4f}')


# %%
