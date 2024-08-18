#%% packages
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# %%
import os
print(os.getcwd())
os.chdir("c:\\Users\\Shuab.Akintola\Desktop\Pytorch_Ultimate\\Datasets")
df = pd.read_csv("heart.csv")
df
df.drop_duplicates(inplace= True)
os.chdir("c:\\Users\\Shuab.Akintola\Desktop\Pytorch_Ultimate")
os.getcwd()
df.head()

# %% separate independent and dependent variable.
X = np.array(df.loc[:, df.columns != 'output'])
print(X)
Y= np.array(df['output'])
print(Y)
print("The shape of X: {} \nThe shape of Y : {}".format(X.shape, Y.shape))
# %%splitting into train and test
x_train, x_test, y_train, y_test = train_test_split(X, Y, 
test_size=0.2, random_state=123)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# %%Scalig the data
scaler = StandardScaler()
x_train_scale = scaler.fit_transform(x_train)
x_test_scale = scaler.transform(x_test)

# %% Creating Neural Network class
class Neural_Network_From_Scratch:
    def __init__(self, LR, X_train, X_test, y_train, y_test):
        self.w = np.random.randn(x_train_scale.shape[1])
        self.b = np.random.randn()
        self.LR = LR
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.L_train = []
        self.L_test = []

# define some helper function
    def activation(self, x):
        return 1/1+np.exp(-x)
    
# derivative of activation function
    def dactivation(self, x):
        return self.activation(x) * (1-self.activation(x))
    
# forward pass
    def forward(self, X):
        hidden_1 = np.dot(X, self.w) + self.b
        activate_1 = self.activation(hidden_1)
        return activate_1

# backward pass   
    def backward_pass(self, X, y_true):
        hidden_1 = np.dot(X, self.w) + self.b
        y_pred = self.forward(hidden_1)
        dL_dpred = 2*(y_pred - y_true)
        dpred_dhidden_1 = self.dactivation(hidden_1)
        dhidden_1_db = 1
        dhidden_1_dw = X
        dL_dw = dL_dpred * dpred_dhidden_1 * dhidden_1_dw 
        dL_db = dL_dpred * dpred_dhidden_1 * dhidden_1_db 
        return y_pred, dL_dw, dL_db



    

# %%
