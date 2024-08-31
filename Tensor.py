#%%Tensor training
import torch
import pandas as pd
import numpy as np
import seaborn as sns
# %%
# create a tensor
x = torch.tensor(5.0)

#%% siomple calculation
y = x + 4
print(y)
# %%
x.requires_grad
# %%
x = torch.tensor(10.0, requires_grad = True)

y = x - 5
print(y.grad)
# %%
def function_y(val):
    return (val-3)* (val-6)* (val-4)

x_range = np.linspace(1,10, 101)
y_range = [function_y(j) for j in x_range]
sns.lineplot(x= x_range, y = y_range)
# %%
y = (x-3)*(x-6)*(x-4)
y.backward()
print(x.grad)
# %%
x = torch.tensor(.0, requires_grad=True)  # Example value for x
y = (x - 3) * (x - 6) * (x - 4)

# Perform backpropagation
y.backward()

# Print the gradient
print(x.grad)
# %%
# Define a tensor with requires_grad=True to track computations
x = torch.tensor(9.0, requires_grad=True)

# Define a simple computation
y = x ** 2 + 54*x # y = x^2

# Compute the gradient
y.backward()

# The gradient is stored in x.grad
print(x.grad) 
# %%
x = torch.tensor(.0, requires_grad= True)
y = x**3
z = 2*y 
z.backward()
print(x.grad)
# %%
x11 = torch.tensor(2.0, requires_grad=True)
x21 = torch.tensor(3.0, requires_grad= True)
x12 = 5*x11 -3*x21
x22 = 2*x11**2 + 2*x21
y = 4*x12 + 3*x22
y.backward()
print(x11.grad)
print(x21.grad)
# %%
cars_file = 'https://gist.githubusercontent.com/noamross/e5d3e859aa0c794be10b/raw/b999fb4425b54c63cab088c0ce2c0d6ce961a563/cars.csv'
cars = pd.read_csv(cars_file)
cars.head()
# %%
cars.to_csv('car_data.csv', index=False)
# %%
sns.scatterplot(x='wt', y='mpg', data=cars)
sns.regplot(x='wt', y='mpg', data=cars)
# %%
