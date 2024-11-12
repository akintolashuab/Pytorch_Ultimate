#%%Import library
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split 
from sklearn.datasets import make_multilabel_classification
import matplotlib.pyplot as plt
import graphlib


# %%
X, y = make_multilabel_classification(n_samples=10000, n_features= 10, n_classes= 3, n_labels= 2)
# %%
X_torch = torch.tensor(X, dtype=torch.float32)
y_torch = torch.tensor(y, dtype= torch.long)
# %% create train and test data
X_train, X_test, y_train, y_test = train_test_split(X_torch, y_torch, test_size=0.2, 
                    random_state=42, shuffle= True)
# %% create dataset class
class MultiLabelData(Dataset):
    def __init__(self, X, y):
        super().__init__()
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# %%create instances of test and train data
train_data_class = MultiLabelData(X_train, y_train)
test_data_class = MultiLabelData(X_test, y_test)

# %%
train_loader = DataLoader(dataset=train_data_class, batch_size=32, shuffle= True)
# %% create nn.Module inheritance and forward pass
class MultiLabelClassifier(nn.Module):
    def __init__(self, NUM_FEATURES, NUM_CLASSES, HIDDEN_FEATURES):
        super(). __init__ ()
        self.lin1 = nn.Linear(NUM_FEATURES,  HIDDEN_FEATURES)
        self.lin2 = nn.Linear(HIDDEN_FEATURES, NUM_CLASSES)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.sigmoid(x)
        return x
    
#%%
NUM_FEATURES = train_data_class.X.shape[1] 
NUM_CLASSES = train_data_class.y.shape[1] 
HIDDEN_FEATURES = 30
model = MultiLabelClassifier(NUM_FEATURES, NUM_CLASSES, HIDDEN_FEATURES)

# %% loss function
loss_ftn = nn.BCEWithLogitsLoss()
lr = 0.02
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# %% creating training loop
losses = [] 
NUM_EPOCHS = 300
for epoch in range(NUM_EPOCHS):
    for i, (X,y) in enumerate(train_loader):
        #convert y to y.long so that the pytorch will be able to see the indexes of the y datapoints
        y = y.float()
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

# %% model testing
with torch.no_grad():
    y_test_pred = model(X_test).round()
    print(y_test_pred)

# %%
print(accuracy_score(y_test_pred, y_test))


# %%
