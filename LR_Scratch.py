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
    return np.dot(2*x, y_hat-y).mean()

print(f'Prediction before training: f(6) = {forward_pass(6):.3f}')

#Training

lr = 0.01
iteration = 100

for i in range(iteration):
    y_pred = forward_pass(X)
    l = loss(Y, y_pred)
    dw = gradient(X, Y, y_pred)
    w-=lr*dw

    if i%1 == 0:
        print(f'Iteration {i}: w = {w:.3f}: Loss = {l:.8f}')

print(f'Prediction after training: f(6) = {forward_pass(6):.3f}')




