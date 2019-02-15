import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import math

def tanh(x):
    return np.tanh(x)
def dtanh(x):
    return 1 - np.power(np.tanh(x), 2)

def relu(x):
    return np.maximum(0, x)
def drelu(x):
    return 1*(x > 0) + 0.5*(x == 0)

def leakyrelu(x, alpha):
    return x*(x > 0) + alpha*x*(x < 0)
def dleakyrelu(x, alpha):
    return 1*(x > 0) + alpha*(x < 0) + (1+alpha)/2*(x==0)
    
def orelu(x):
    y = relu(x)
    toChange = np.sum(y == 0) == x.shape[-1]
    y[toChange, :] = relu(-x[toChange, :])
    return y
def dorelu(x):
    result = np.zeros(x.shape)
    result[x > 0] = 1
    y = relu(x)
    toChange = np.sum(y == 0, axis=1) == x.shape[-1]
    result[toChange, :] = np.array([-1, -1])
    return result

# Note: this returns a scalar    
def basisTest(x):
    xVal = np.sqrt(np.sum(np.power(np.abs(x), 2))) # could subtract c here
    f = (1-xVal)*(xVal >= 0 and xVal < 1)
    return f*np.ones(x.shape)
   
def dbasisTest(x):
    xVal = np.sqrt(np.sum(np.power(np.abs(x), 2))) # could subtract c here
    f = 2*x*(xVal >= 0 and xVal < 1)
    return -f

class NeuralNet:
    def __init__(self, nonlinear_func, nonlinear_func_deriv, nhiddens, eps=0.001):
        self.func = nonlinear_func
        self.func_deriv = nonlinear_func_deriv
        self.eps = eps
        self.hiddens = []
        for i in range(len(nhiddens)-1):
            self.hiddens.append(0.1*(2*np.random.uniform(size=(nhiddens[i]+1,nhiddens[i+1]))-1))
        self.W = 0.1*(2*np.random.uniform(size=(nhiddens[-1]+1,1))-1)

    def train(self, data_set, epochs, batch_size):
        for i in range(epochs):
            data = data_set[np.random.randint(data_set.shape[0], size=batch_size), :]

            inputs = [data[:, :-1]]
            for i in range(len(self.hiddens)):
                Z = self.func(inputs[-1].dot(self.hiddens[i][1:, :]) + self.hiddens[i][0, :])
                inputs.append(Z)
            Y = inputs[-1].dot(self.W[1:,:]) + self.W[0,:]
            error = data[:, -1:] - Y

            self.W[1:, :] += self.eps*inputs[-1].T.dot(error)
            self.W[0, :] += self.eps*np.sum(error, 0)

            deltaPlus = (error.dot(self.W[1:,:].T))*self.func_deriv(inputs[-1])

            for i in range(len(self.hiddens)-1, -1, -1):
                self.hiddens[i][1:, :] += self.eps*inputs[i].T.dot(deltaPlus)
                self.hiddens[i][0, :] += self.eps*np.sum(deltaPlus, 0)

                deltaPlus = (deltaPlus.dot(self.hiddens[i][1:, :].T))*self.func_deriv(inputs[i])

    def predict(self, data):
        # intermediate = [data]
        # Z1 = self.func(data.dot(self.U[1:,:]) + self.U[0,:])
        # intermediate.append(Z1)
        # Z2 = self.func(Z1.dot(self.V[1:,:]) + self.V[0,:])
        # intermediate.append(Z2)
        # Y = Z2.dot(self.W[1:,:]) + self.W[0,:]
        # intermediate.append(Y)

        inputs = [data]
        for i in range(len(self.hiddens)):
            Z = self.func(inputs[-1].dot(self.hiddens[i][1:, :]) + self.hiddens[i][0, :])
            inputs.append(Z)

        #Z1 = self.func(data_set[:, :-1].dot(self.U[1:,:]) + self.U[0,:])
        #Z2 = self.func(Z1.dot(self.V[1:,:]) + self.V[0,:])
        Y = inputs[-1].dot(self.W[1:,:]) + self.W[0,:]
        inputs.append(Y)
        return 2*(inputs[-1] > 0) - 1, inputs
