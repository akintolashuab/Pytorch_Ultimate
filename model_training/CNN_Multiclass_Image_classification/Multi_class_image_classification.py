#%%Import Packages
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
import torchvision
from torchvision import transforms as transforms


#%%seting up Image transformation.
transform = transforms.Compose([transforms.Resize((50,50)),
                                transforms.Grayscale(num_output_channels=1),
                                transforms.ToTensor(),
                                transforms.Normalize(.5, .5)])

batch_size = 6
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

#%%
CLASSES = ['affenpinscher', 'akita', 'corgi']

# %%
class MultiClassificationNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,6,3) # 6, 48, 48
        self.pool = nn.MaxPool2d(2,2) #6, 24, 24
        self.conv2 = nn.Conv2d(6, 16, 3) # 16, 22, 22
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16*11*11, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, len(CLASSES))
        self.relu = nn.ReLU()
        self.sofmax = nn.LogSoftmax()

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
        x = self.sofmax(x)
        return x

# %%training loop
num_epoch = 20
LR = 0.001
model = MultiClassificationNet()
loss_ftn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = LR)
for epoch in range(num_epoch):
    for i, data in enumerate(train_data_loader, 0):
        
        inputs, label = data

        # Zero grad
        optimizer.zero_grad()

        #prediction
        output = model(inputs)

        #calculate loss
        loss = loss_ftn(output, label)

        #backward function
        loss.backward()

        #Update weight
        optimizer.step()
    
        if i%2 == 0:
            print(f'Epoch {epoch}/ {num_epoch}, step = {i}/{len(train_data_loader)},'
            f'Loss: {loss.item():.4f}')


# %%
y_test = []
y_tes_pred = []
for i, data in enumerate(test_data_loader, 0):
    inputss, y_test_tempo = data
    with torch.no_grad():
        y_test_hat_temp = model(inputss).round()
    y_test.extend(y_test_tempo.numpy())
    y_tes_pred.extend(y_test_hat_temp.numpy())

acc = accuracy_score(y_test, np.argmax(y_tes_pred, axis = 1))
conf_Matrix = confusion_matrix(y_test, np.argmax(y_tes_pred, axis = 1))
print(f'The accuracy of our model is {round(acc * 100, 2)}%')
conf_Matrix

# %%

