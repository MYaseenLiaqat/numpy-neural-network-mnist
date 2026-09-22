import numpy as np


class DenseLayer:

    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros((1, output_size))

    def forward(self, X):
        self.input = X
        return X @ self.weights + self.biases

    def backward(self, dZ):
        batch_size = self.input.shape[0]

        self.dweights = self.input.T @ dZ / batch_size
        self.dbiases = np.sum(dZ, axis=0, keepdims=True) / batch_size

        dX = dZ @ self.weights.T

        return dX