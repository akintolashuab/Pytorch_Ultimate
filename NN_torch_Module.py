# %%
import torch
import torch.nn as nn

X = torch.tensor([[1],[2],[3],[4],[5]], dtype=torch.float32)
Y = torch.tensor([[1],[4],[9],[16],[25]], dtype=torch.float32)
X_test = torch.tensor([6], dtype=torch.float32)
n_sample, n_features = X.shape
input_size = n_features
output_size = n_features

model = nn.Linear(input_size, output_size)

lr = 0.04
iteration = 200

loss = nn.MSELoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr = lr)

print(f'Prediction before training: f(6) = {model(X_test).item()}')

#Training



for i in range(iteration):
    y_pred = model(X)
    l = loss(Y, y_pred)

    #Gradient dl/dw
    l.backward() #dl/dw

    #update weight
    optimizer.step()

    #Zero Gradient:
    optimizer.zero_grad()
    [w, b] = model.parameters()
    
    if i%10 == 0:
        print(f'Iteration {i}: w = {w[0][0].item():.4f}: b = {b[0].item():.4f}: Loss = {l:.8f}')
    

print(f'Prediction after training: f(6) = {model(X_test).item()}')
# %%

  
