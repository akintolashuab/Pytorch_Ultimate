# %%
import torch 

X = torch.tensor([1,2,3,4,5])
Y = torch.tensor([2,4,6,8,10])

# f = 2*x
# y_pred = w*x

# initialize weight
w = torch.tensor(0.0, requires_grad= True)

def forward_pass(x):
    return w*x

def loss(y, y_hat):
    return ((y-y_hat)**2).mean()

print(f'Prediction before training: f(6) = {forward_pass(6):.3f}')

#Training

lr = 0.01
iteration = 100

for i in range(iteration):
    y_pred = forward_pass(X)
    l = loss(Y, y_pred)

    #Gradient dl/dw
    l.backward() #dl/dw

    #update weight
    with torch.no_grad():
        w-=lr*w.grad

    #Zero Gradient:
    w.grad.zero_()

    if i%10 == 0:
        print(f'Iteration {i}: w = {w:.3f}: Loss = {l:.8f}')

print(f'Prediction after training: f(6) = {forward_pass(6):.3f}')
# %%
