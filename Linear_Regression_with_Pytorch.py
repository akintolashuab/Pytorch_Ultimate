# %%
# Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn import datasets
from sklearn.model_selection import train_test_split as tts
import seaborn as sns

# %%
# Data Preparation
X_numpy, Y_numpy = datasets.make_regression(n_samples=1000, n_features=1, noise=20, random_state=1)
X = torch.from_numpy(X_numpy.astype(np.float32))
Y = torch.from_numpy(Y_numpy.astype(np.float32))
y = Y.view(-1, 1)  # Making sure Y is 2D with shape [n_samples, 1]

# %%
x_train, x_test, y_train, y_test = tts(X, y, test_size=0.2, shuffle=True, random_state=30)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# %%
n_samples, n_features = X.shape
input_size = n_features
output_size = 1

# %%
# Model Specification
model = nn.Linear(input_size, output_size)
learning_rate = 0.1
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# %%
# Training Loop
iterations = 1000
losses = []
for epoch in range(iterations):
    # forward pass
    y_predicted = model(x_train)

    # Loss
    l = criterion(y_predicted, y_train)

    # Backward
    l.backward()

    optimizer.step()
    optimizer.zero_grad()

    # Access weight and bias
    [w, b] = model.parameters()
    
    if epoch % 20 == 0:
        print(f'Iteration {epoch}: w = {w[0][0].item():.4f}, \
              b = {b[0].item():.4f}, loss = {l.item():.8f}')  # Correct use of l.item() for scalar loss
    
    losses.append(l.item())  # Storing the scalar value of the loss

# %%
# Plot the loss
loss_df = pd.DataFrame(losses, columns=['Model_Loss'])
sns.lineplot(data=loss_df, x=loss_df.index, y='Model_Loss')
plt.show()
# %%
x_test_numpy = x_test.detach().numpy()
y_test_numpy = y_test.detach().numpy()
predicted = model(x_test).detach().numpy()
plt.plot(x_test_numpy, predicted, 'b')
plt.plot(x_test_numpy, y_test_numpy, 'ro')
plt.show()
# %%
df = pd.DataFrame({"True_Value": y_test_numpy.flatten(), 'Prediction':predicted.flatten()})
df
# %%
predicted
# %%
