#%%import packages
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt
import graphlib


#%%import dataset
iris = load_iris()
X = iris.data
y = iris.target

#%%train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                        test_size= 0.2, random_state=42)

X_test = X_test.astype('float32')
X_train = X_train.astype('float32')

# %%S
X = torch.from_numpy(X_train)
y = torch.tensor(y_train)
X
# %% Create Dataset and Dataloader class
class IrisData(Dataset):
    def __init__(self, X, y):
        super().__init__()
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
Iris_data = IrisData(X_train, y_train)
train_loader = DataLoader(dataset = Iris_data, batch_size=32, shuffle= True)

#%% Create inheritance from nn module
class MultiCLassNet(nn.Module):
    #lin1, lin2, logsoftmax
    def __init__(self, NUM_FEATURES, NUM_CLASSES, HIDDEN_FEATURES):
        super().__init__()
        self.lin1 = nn.Linear(NUM_FEATURES, HIDDEN_FEATURES)
        self.lin2 = nn.Linear(HIDDEN_FEATURES, NUM_CLASSES)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.lin1(x)
        x = torch.relu(x)
        x = self.lin2(x)
        x = self.log_softmax(x)
        return x
    
#%%
NUM_FEATURES = Iris_data.X.shape[1]
NUM_CLASSES = len(np.unique(Iris_data.y))
HIDDEN_FEATURES = 6
print(NUM_FEATURES)
print(NUM_CLASSES)
model = MultiCLassNet(NUM_FEATURES, NUM_CLASSES, HIDDEN_FEATURES)

# %%Specifying loss function.
loss_ftn = nn.CrossEntropyLoss()


# optimizer
LR =0.02
optimizer = torch.optim.SGD(model.parameters(), lr = LR)

# %% model iteration.
losses =[]
slope =[]
bias = []
num_epoch = 1000
for epoch in range(num_epoch):
    for X, y in train_loader:
        y = y.long()
        #set the gradients to zero
        optimizer.zero_grad()

        #forward pass
        y_pred = model(X)
    
        loss = loss_ftn(y_pred, y)
        loss.backward()

        #update gradient
        optimizer.step()
        
    losses.append(loss.item())
    if epoch % 100 == 0:
        print("Epoch: {}, Loss : {:.4f}".format(epoch, loss.item()))

# %% visualize the loss
sns.lineplot(x = range(num_epoch), y =losses)

# %%
X_test_torch = torch.from_numpy(X_test)
with torch.no_grad():
    y_test_log = model(X_test_torch)
    y_test_pred = torch.max(y_test_log.data, 1)

# %%
y_test
# %%
y_test_pred
# %%
accuracy_score(y_test, y_test_pred.indices)
# %%
confusion_matrix(y_test, y_test_pred.indices)
# %%
