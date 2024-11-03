#%% Import of Packages
import numpy as np
import pandas as pd
import torch
from torch import nn 
import seaborn as sns
import matplotlib.pyplot as plt

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
y_list 
# %%
X = torch.from_numpy(X_numpy)
y = torch.tensor(y_list)
y
X
# %%
w = torch.rand(1, requires_grad= True, dtype= torch.float32)
b = torch.rand(1, requires_grad= True, dtype= torch.float32)
num_epoch = 1000
learning_rate = 0.001
loss_values_1 = []

# forward pass
for epoch in range(num_epoch):
    for i in range(len(X)):

        #forward pass
        y_pred = X[i] * w + b

        #loss
        loss = torch.pow(y_pred - y[i], 2)

        #backward pass
        loss.backward()

        #get the loss values
        loss_value = loss.data[0]
        

        #weight update
        with torch.no_grad():
            w -= w.grad * learning_rate
            b -= b.grad * learning_rate
            w.grad.zero_()
            b.grad.zero_()
    loss_values_1.append(loss.item())
    print(loss_value)
sns.lineplot(x=range(len(loss_values_1)), y=loss_values_1)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Loss over Iterations")
plt.show()



# %%
