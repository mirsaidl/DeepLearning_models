import numpy as np
import math

def normalize(x):
    mean = np.mean(x)
    std_dev = np.std(x)
    z = (x - mean) / std_dev
    return z

def sigmoid(z):
    z_clipped = np.clip(z, -500, 500)  # Clip values to the range [-500, 500]
    return 1 / (1 + np.exp(-z_clipped))

def compute_loss(y_pred, y):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    loss = 0
    for i in range(y.shape[0]):
        loss += - (y[i] * np.log(y_pred[i]) + (1 - y[i]) * np.log(1 - y_pred[i])) 
    return loss

def backward_prop():
    pass

def fit(x, y, lr,epochs):
    m, n = x.shape
    w = np.zeros((n,))
    b = 0 #np.zeros(n,1)
    for i in range(epochs):
        z = np.dot(x, w) + b
        a = sigmoid(z)
        
        dz = a - y
        dw = np.dot(x.T, dz) / m
        db = np.sum(dz) / m
        
        w -= lr * dw
        b -= lr * db
        
        if i % math.ceil(epochs/10) == 0:
            print(f"Iteration {i:4d}: Cost {compute_loss(a, y):4d}")
            
    return w, b

def predict(x,w,b):
    m = x.shape[0]
    y_hat = []
    for i in range(m):
        z = np.dot(x[i],w) + b
        a = sigmoid(z)
        if a > 0.5:
            y_hat.append(1)
        else:
            y_hat.append(0)
            
    return np.array(y_hat)

def accuracy(y_pred, y):
    return  (sum(y_pred == y) / y.shape[0]) * 100
