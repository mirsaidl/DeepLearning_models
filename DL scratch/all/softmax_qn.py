import numpy as np
import math

# Functions
def normalize(x):
    mean = np.mean(x)
    std_dev = np.std(x)
    z = (x - mean) / std_dev
    return z

def one_hot_encode(labels, num_classes):
    encoded_labels = np.zeros((len(labels), num_classes))
    for i, label in enumerate(labels):
        encoded_labels[i, label] = 1
    return encoded_labels

def linear(x,w,b):
    return (np.dot(x,w) + b)

def softmax(z):
    ezz = np.exp(z - np.max(z, axis=1, keepdims=True))  # Subtract max for numerical stability
    ezz_sum = np.sum(ezz, axis=1, keepdims=True)
    a = ezz / ezz_sum
    return a

def softmax_derivative(x):
    s = softmax(x)
    return s * (1 - s)

def relu(z):
    return np.maximum(0,z)

def relu_derivative(z):
     return np.where(z > 0, 1, 0)

def sparse_crossentropy(y_pred, y):
    m = y_pred.shape[0]
    y_pred = np.clip(y_pred, 1e-15, 1e-15)
    # Convert y to integer type
    y_int = y.astype(int)
    loss = -np.sum(np.log(y_pred[range(m), y_int])) / m
    return loss

def init_params(x, hidden_units, output_units):
    np.random.seed(30)
    input_units = x.shape[1]
    w1 = np.random.randn(input_units, hidden_units) * 0.01
    b1 = np.zeros((hidden_units, 1))
    w2 = np.random.randn(hidden_units, output_units) * 0.01
    b2 = np.zeros((output_units, 1))
    
    return w1, b1, w2, b2
    
def back_prop(x,y,a1,a2,w2,z1,b1,b2):
    m = x.shape[1]
    
    if y.ndim == 1:
        y = y.reshape(-1,1)
    
    dz2 = a2 - y
    dw2 = np.dot(a1.T, dz2) / m
    db2 = np.sum(dz2.T, axis=1, keepdims=True) / m
    dz1 = np.dot(dz2, w2.T) * (relu_derivative(z1))
    dw1 = np.dot(x.T, dz1) / m
    db1 = np.sum(dz1.T, axis=1, keepdims=True) / m
    
    return dw1, db1, dw2, db2

def predict(x, w1, b1, w2, b2):
    z1 = np.dot(x, w1.T) + b1.reshape(1,w1.shape[0])
    a1 = relu(z1)
    z2 = np.dot(a1, w2.T) + b2
    a2 = softmax(z2)
    
    yhat = np.argmax(a2, axis=1)
    
    return yhat.T

def accuracy(yhat, y):
    return  (np.sum(yhat == y) / y.shape[0]) * 100

def fit(x, y, hidden_units, learning_rate, epochs):
    w1, b1, w2, b2 = init_params(x, hidden_units, output_units=10)
    
    for i in range(epochs+1):
        z1 = linear(x,w1,b1.T)
        a1 = relu(z1)
        z2 = linear(a1,w2,b2.T)
        a2 = softmax(z2)
        
        print(a2.shape, z2.shape, a1.shape)
        
        dw1, db1, dw2, db2 = back_prop(x, y, a1, a2, w2, z1, b1, b2)
        
        w1 -= learning_rate * dw1
        b1 -= learning_rate * db1
        w2 -= learning_rate * dw2 
        b2 -= learning_rate * db2
        
        if i % math.ceil(epochs/10) == 0:
            print(f"Iteration: {i} Loss: {sparse_crossentropy(a2, y)}")
    
    return w1, b1, w2, b2