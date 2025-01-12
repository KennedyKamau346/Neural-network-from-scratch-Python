import numpy as np

inputs = [1, 2, 3, 2.5]      # vector

weights = [[0.2, 0.8, -0.5, 1.0],          # matrix containing vectors
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]
           ]

biases = [2, 3, 0.5]
bias1 = 2
bias2 = 3
bias3 = 0.5

output = np.dot(weights, inputs) + biases
print(output)