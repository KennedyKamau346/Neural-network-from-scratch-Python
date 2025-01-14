import numpy as np

np.random.seed(0)

X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # * 0.10 is to get weights between -1 and 1 and the shape is n inputs by n neurons and is the opposite of
        # the previous parts. We do this to avoid transposing every time we do a forward pass
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))  # the shape is 1 by however many neurons we have

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


layer1 = Layer_Dense(4, 5)
layer2 = Layer_Dense(5,2)

layer1.forward(X)
print(layer1.output)
layer2.forward(layer1.output)
print(layer2.output)