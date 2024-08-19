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
        self.w = np.random.randn(X_train.shape[1])
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
        return 1/(1+np.exp(-x))
    
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

# Optimizer
    def optimizer(self, dL_dw, dL_db):
        self.w = self.w - dL_dw * self.LR
        self.b = self.b - dL_db * self.LR
        return f"the best params are: weight = {self.w} \nbias = {self.b}"

# training the Data
    def train(self, ITERATIONS):
        for i in range(ITERATIONS):
            #define some random position
            random_pos = np.random.randint(len(self.X_train))

            #forward pass
            y_train_true = self.y_train[random_pos]
            y_train_pred = self.forward(self.X_train[random_pos])

            #Calculate training losses
            L = np.square(y_train_true - y_train_pred)
            self.L_train.append(L)

            #backward pass
            y_pred, dL_dw, dL_db = self.backward_pass(self.X_train[random_pos], 
                                                 y_train[random_pos])
            
            #Optimizer
            self.optimizer(dL_dw, dL_db)

            L_sum = 0
            for j in range(len(self.X_test)):
                y_true = self.y_test[j]
                y_pred = self.forward(self.X_test[j])
                L_sum += np.square(y_true - y_pred)
            self.L_test.append(L_sum)
        return "Training Successful"
    
#%% Model Instance and Model Training    
LR = 0.1
ITERATIONS = 1000

nn = Neural_Network_From_Scratch(LR = LR, X_train = x_train_scale, 
        X_test = x_test_scale, y_train = y_train, y_test = y_test)
nn.train(ITERATIONS = ITERATIONS)
#print(nn.L_train)
pd.DataFrame(nn.L_test)
#print(len(nn.L_test))
#pd.DataFrame(nn.L_train)

# %% Full batch gradient descent
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Neural Network class with full-batch Gradient Descent
class Neural_Network_From_Scratch:
    def __init__(self, LR, X_train, X_test, y_train, y_test):
        # Initialize weights, bias, and other parameters
        self.w = np.random.randn(X_train.shape[1])
        self.b = np.random.randn()
        self.LR = LR
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.L_train = []
        self.L_test = []

    # Sigmoid activation function
    def activation(self, x):
        return 1 / (1 + np.exp(-x))

    # Derivative of the sigmoid activation function
    def dactivation(self, x):
        return self.activation(x) * (1 - self.activation(x))

    # Forward pass
    def forward(self, X):
        z = np.dot(X, self.w) + self.b
        return self.activation(z)

    # Backward pass to calculate gradients
    def backward_pass(self):
        # Forward pass to get predictions
        y_pred = self.forward(self.X_train)
        
        # Calculate gradients
        dL_dpred = 2 * (y_pred - self.y_train) / len(self.y_train)
        dpred_dz = self.dactivation(np.dot(self.X_train, self.w) + self.b)
        dz_dw = self.X_train
        dz_db = 1

        # Gradients with respect to weights and bias
        dL_dw = np.dot(dz_dw.T, dL_dpred * dpred_dz)
        dL_db = np.sum(dL_dpred * dpred_dz * dz_db)

        return dL_dw, dL_db

    # Update weights and bias using gradient descent
    def optimizer(self, dL_dw, dL_db):
        self.w -= self.LR * dL_dw
        self.b -= self.LR * dL_db

    # Training the model using full-batch Gradient Descent
    def train(self, ITERATIONS):
        for i in range(ITERATIONS):
            # Forward pass and calculate gradients
            dL_dw, dL_db = self.backward_pass()

            # Update weights and bias
            self.optimizer(dL_dw, dL_db)

            # Calculate and store the loss for the training set
            y_train_pred = self.forward(self.X_train)
            L_train = np.mean(np.square(self.y_train - y_train_pred))
            self.L_train.append(L_train)

            # Calculate and store the loss for the test set
            y_test_pred = self.forward(self.X_test)
            L_test = np.mean(np.square(self.y_test - y_test_pred))
            self.L_test.append(L_test)
        print("Training Successful")
        df_result = pd.DataFrame({ "Y_test": self.y_test,
                    "Y_Prediction": np.round(y_test_pred) })

        return df_result

# Data preparation (Example data, assuming df is your dataframe)
X = np.array(df.loc[:, df.columns != 'output'])
Y = np.array(df['output'])

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=123)

# Scale the data
scaler = StandardScaler()
x_train_scale = scaler.fit_transform(x_train)
x_test_scale = scaler.transform(x_test)

# Model training
LR = 0.1
ITERATIONS = 1000 

nn = Neural_Network_From_Scratch(LR=LR, X_train=x_train_scale, X_test=x_test_scale, y_train=y_train, y_test=y_test)
nn.train(ITERATIONS=ITERATIONS)

# Print the training and test losses
train_loss_df = pd.DataFrame(nn.L_train, columns=['Training Loss'])
test_loss_df = pd.DataFrame(nn.L_test, columns=['Test Loss'])
print(train_loss_df)
print(test_loss_df)

# %%
