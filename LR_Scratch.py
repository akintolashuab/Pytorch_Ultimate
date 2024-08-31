# %%
import numpy as np

X = np.array([1,2,3,4,5])
Y = np.array([2,4,6,8,10])

# f = 2*x
# y_pred = w*x

# initialize weight
w = 0.0

def forward_pass(x):
    return w*x

def loss(y, y_hat):
    return ((y-y_hat)**2).mean()

def gradient(x, y, y_hat):
    return np.dot(2*2, y_hat-y).mean()

print(f'prediction before training: f(6) = {forward_pass(5):.3f}')

#Training





# %%
