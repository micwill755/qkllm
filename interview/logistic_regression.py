import numpy as np
import pandas as pd

# Q3 implement Logistic regression from scratch

# load data
data = np.genfromtxt('data/toydata.txt', delimiter='\t')
X = data[:, :2]
y = data[:, 2]

perc_70 = int(len(data) * 0.7)

X_train, X_test = X[:perc_70], X[perc_70:]
y_train, y_test = y[:perc_70], y[perc_70:]

# normalize X
mu, sigma = X_train.mean(), X_train.std()
X_train = (X_train - mu) / sigma
X_test = (X_test - mu) / sigma

# create logistic regression class
class LogisticRegression:
    def __init__(self, dim):
        self.weights = np.random.randn(dim)
        self.bias = 0

    def forward(self, x):
        o = x @ self.weights + self.bias
        return self._sigmoid(o)

    def backward(self, x, yhat, y, learning_rate):
        dw = x.T @ (yhat - y) / len(y)
        db = np.mean(yhat - y)
        self.weights -= learning_rate * dw
        self.bias -= learning_rate * db

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

# cross entropy loss instead of mse
def cross_entropy_loss(yhat, y):
    eps = 1e-15 # in case yhat is 0
    yhat = np.clip(yhat, 1e-15, 1-eps)
    return -np.mean(y * np.log(yhat))

# batch gradient descent
def bgd(model, x, y, epochs, learning_rate=0.01):
    cost = []
    for e in range(epochs):
        yhat = model.forward(x)
        loss = cross_entropy_loss(yhat, y)
        cost.append(loss)
        model.backward(x, yhat, y, learning_rate)

        if e % 10 == 0:
            print(f'epoch {e}, loss {loss}')

    return cost

model = LogisticRegression(X.shape[1])
bgd(model, X_train, y_train, 100, 0.05)