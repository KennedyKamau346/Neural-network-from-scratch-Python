# this shows the rectified linear unit activation function
import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]

X, y = spiral_data(100, 3)
'''
inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
outputs = []

# ReLU activation function. For input > 0 output is input but for input 0 or negative output will be 0 
for i in inputs:                     // one way to do it
    if i > 0:
        output.append(i)
    elif i <= 0:
        output.append(0)

print(output)                        
outputs = []                           
for i in inputs:                        // another way to do it
    output.append(max(0, i))

print(outputs)
'''


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.output = None
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


layer1 = Layer_Dense(2, 5)
activation1 = Activation_ReLU()

layer1.forward(X)

print(layer1.output)
activation1.forward(layer1.output)
print(activation1.output)
