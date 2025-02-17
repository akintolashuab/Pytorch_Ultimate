#%%import packages

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from skorch import NeuralNetRegressor
from sklearn.model_selection import GridSearchCV
import graphlib

#%%import dataset
data = pd.read_csv("C:\\Users\\Shuab.Akintola\\Desktop\\Pytorch_Ultimate\\car_data.csv")
data.head()
data.shape

#%%Visualize the data
sns.scatterplot(x= "wt", y = "mpg", data=data)
sns.regplot(x= "wt", y = "mpg", data=data)

# %%
X_list = data.wt.values
X_numpy = np.array(X_list, dtype=np.float32).reshape(-1,1)
X_numpy
y_list = data.mpg.values.tolist()
y_numpy = np.array(y_list, dtype=np.float32).reshape(-1,1)
# %%S
X = torch.from_numpy(X_numpy)
y = torch.tensor(y_numpy)
y
X
# %% Create Dataset and Dataloader class
class LinearRegressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
train_loader = DataLoader(dataset = LinearRegressionDataset(X_numpy, y_numpy), batch_size=2)

#%% Create inheritance from nn module
class LinearRegressionTorch(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegressionTorch, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        out = self.linear(x)
        return out
input_dim =1
output_dim =1
model = LinearRegressionTorch(input_dim, output_dim)

# %%Specifying loss function.
loss_ftn = nn.MSELoss()


# optimizer
LR =0.02
optimizer = torch.optim.SGD(model.parameters(), lr = LR)

# %% model iteration.
net = NeuralNetRegressor(model, max_epochs = 1000, 
                         lr =LR, iterator_train__shuffle = True )
# %%
net.set_params(train_split = False, verbose = 0)
# %%
params = {'lr' : [0.02, 0.05, 0.08],
           'max_epochs': [20, 200, 500],
           'batch_size': [2, 6, 8]}

gs = GridSearchCV(net, params, scoring= 'r2', verbose =2, cv = 5)
gs.fit(X, y)
print(f'best score : {gs.best_score_} \nbest params: {gs.best_params_}')

# %%
