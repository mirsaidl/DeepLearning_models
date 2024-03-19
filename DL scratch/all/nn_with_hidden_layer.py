import math
import numpy as np

# Neural Network with hidden units
def normalize(x):
    mean = np.mean(x)
    std_dev = np.std(x)
    z = (x - mean) / std_dev
    return z

def softmax(z):
    N = len(z)
    a = np.zeros(N)
    ezz_sum = 0
    for k in range(N):
        ezz_sum += np.exp(z[k])
    for j in range(N):
        a[j] = np.exp(z[j]) / ezz_sum
    return a

def softmax_derivative(x):
    s = softmax(x)
    return s * (1 - s)

def sigmoid(z):
    z_clipped = np.clip(z, -500, 500)  # Clip values to the range [-500, 500]
    return 1 / (1 + np.exp(-z_clipped))

def sigmoid_derivative(z):
    gz = sigmoid(z)
    return gz * (1 - gz)

def linear(x,w,b):
    return (w * x) + b

def relu(z):
    return np.maximum(0,z)

def relu_derivative(z):
     return np.where(z > 0, 1, 0)

def tanh_derivative(z):
    return 1 - (tanh(z) ** 2)

def leaky_relu(z):
    return np.maximum(0.01 * z, z)

def tanh(z):
    return np.tanh(z)

def loss_sigmoid(y_pred, y):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    loss = np.mean(- (y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))) 
    return loss

def sparse_crossentropy(y_pred, y):
    m = y_pred.shape[0]
    y_pred = np.clip(y_pred, 1e-15, 1e-15)
    loss = -np.sum(np.log(y_pred[range(m), y])) / m
    
    return loss

def init_params(x, hidden_units, output_units):
    np.random.seed(30)
    input_units = x.shape[1]
    w1 = np.random.randn(hidden_units, input_units)
    b1 = np.zeros((hidden_units,  output_units))
    w2 = np.random.randn(output_units, hidden_units)
    b2 = np.zeros((output_units, 1))
    
    return w1, b1, w2, b2
    
def back_prop(x,y,a1,a2,w2,z1):
    m = x.shape[1]
    
    dz2 = a2 - y # y stacked horizontally
    dw2 = np.dot(dz2 ,a1.T) / m
    db2 = (np.sum(dz2, axis=1, keepdims=True)) / m
    dz1 = np.dot(w2.T,dz2) * relu_derivative(z1)
    dw1 = np.dot(dz1 ,x) / m
    db1 = (np.sum(dz1, axis=1, keepdims=True)) / m
    
    return dw1, db1, dw2, db2

def forward_prop(x, y, hidden_units, learning_rate, epochs):
    w1, b1, w2, b2 = init_params(x, hidden_units, output_units=1)
    
    for i in range(epochs+1):
        z1 = np.dot(w1,x.T) + b1
        a1 = relu(z1)
        z2 = np.dot(w2, a1) + b2
        a2 = sigmoid(z2)
        
        dw1, db1, dw2, db2 = back_prop(x, y, a1, a2, w2, z1)
        
        w1 -= learning_rate * dw1
        b1 -= learning_rate * db1
        w2 -= learning_rate * dw2 
        b2 -= learning_rate * db2
        
        if i % math.ceil(epochs/10) == 0:
            print(f"Iteration: {i} Loss: {loss_sigmoid(a2, y)}")
    
    return w1, b1, w2, b2

def predict(x, w1, b1, w2, b2):
    z1 = np.dot(x, w1.T) + b1.reshape(1,w1.shape[0])
    a1 = relu(z1)
    z2 = np.dot(a1, w2.T) + b2
    a2 = sigmoid(z2)
    
    y_hat = (a2 > 0.5).astype(int)
    
    return y_hat.T

def accuracy(yhat, y):
    return  (np.sum(yhat == y) / y.shape[0]) * 100
    
fit = forward_prop

