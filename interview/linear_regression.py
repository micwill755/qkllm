# Q2 implement linear regression with gradient descent

import pandas as pd
import numpy as np

class LinearRegression:
    def __init__(self, dim):
        self.weights = np.random.randn(dim)
        self.bias = 0
    
    def forward(self, x):
        # y = mx + b
        yhat = x @ self.weights + self.bias
        return yhat

    def backward(self, x, yhat, y, learning_rate):
        # calculate error yhat - y
        e = yhat - y

        # derivate of mse d/de (1/n)Σe² = (2/n)e
        dL_de = (2 * e) / len(e)

        # Chain rule: dL/dw = dL/de * de/dyhat * dyhat/dw
        # Since e = yhat - y, de/dyhat = 1
        # Since yhat = x @ w + b, dyhat/dw = x
        dw = x.T @ dL_de  # Weight gradients # (2, 700) @ (700,) → (2,) 
        db = np.mean(dL_de)  # Bias gradient

        self.weights -= learning_rate * dw
        self.bias -= learning_rate * db
        

# step 1: load data from csv
df = pd.read_csv('./data/linreg-data.csv')
#print(df.sample(5))

# step 2: prepare X and y
X = df[['x1', 'x2']]
y = df[['y']]

per_70 = int(len(X) * 0.7)

X_train, X_test = X[:per_70], X[per_70:]
y_train, y_test = y[:per_70], y[per_70:]

print(len(X_train), len(X_test))

# step 3: normalize

# FROM SCRATCH:
'''# calculate means
mus = []
for f in X_train:
    mu = 0
    for d in X_train[f]:
        mu += d
    mus.append(mu / len(X_train[f]))
print(mus)

# calculate standard deviations
sigs = []
for i, f in enumerate(X_train):
    sig = 0
    for d in X_train[f]:
        sig += (d - mus[i]) ** 2
    sigs.append((sig / len(X_train)) ** 0.5)
print(sigs)'''

mu, sig = X_train.mean(), X_train.std()
X_train = (X_train - mu) / sig
X_test = (X_test - mu) / sig

# step 4: implement mse
def mse(yhat, y):
    return np.mean((yhat - y) ** 2)

# step 5: implement stochastic gradient descent
def bgd(model, x, y, epochs, learning_rate=0.01):
    cost = []
    for e in range(epochs):
        # forward pass
        yhat = model.forward(x)
        # calculate loss
        loss = mse(yhat, y)
        cost.append(loss)
        # calculate gradients
        model.backward(x, yhat, y, learning_rate)
        if e % 10 == 0:
            print(f'epoch {e}, loss {loss}')
    return cost

# step 6 instantiate a linear object
model = LinearRegression(X_train.shape[1])
cost = bgd(model, X_train.values, y_train.values.flatten(), epochs=100, learning_rate=0.05)