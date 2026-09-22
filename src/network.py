import numpy as np

from src.layers import DenseLayer
from src.activations import relu, softmax, relu_derivative
from src.losses import cross_entropy_loss


class NeuralNetwork:

    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        self.layer1 = DenseLayer(input_size, hidden_size)
        self.layer2 = DenseLayer(hidden_size, output_size)

    def forward(self, X):
        self.z1 = self.layer1.forward(X)
        self.a1 = relu(self.z1)

        self.z2 = self.layer2.forward(self.a1)
        self.a2 = softmax(self.z2)

        return self.a2

    def backward(self, y_true):
        dz2 = self.a2 - y_true

        da1 = self.layer2.backward(dz2)

        dz1 = da1 * relu_derivative(self.z1)

        self.layer1.backward(dz1)

    def update_parameters(self, learning_rate):
        self.layer1.weights -= learning_rate * self.layer1.dweights
        self.layer1.biases -= learning_rate * self.layer1.dbiases

        self.layer2.weights -= learning_rate * self.layer2.dweights
        self.layer2.biases -= learning_rate * self.layer2.dbiases

    def train(self, X_train, y_train, epochs=5, batch_size=64, learning_rate=0.01):
        num_samples = X_train.shape[0]
        history = {"loss": []}

        for epoch in range(epochs):
            indices = np.random.permutation(num_samples)

            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            epoch_loss = 0.0

            for start in range(0, num_samples, batch_size):
                end = start + batch_size

                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                predictions = self.forward(X_batch)

                loss = cross_entropy_loss(y_batch, predictions)
                epoch_loss += loss

                self.backward(y_batch)

                self.update_parameters(learning_rate)

            average_loss = epoch_loss / np.ceil(num_samples / batch_size)

            history["loss"].append(average_loss)

            print(
                f"Epoch {epoch + 1}/{epochs} - "
                f"Loss: {average_loss:.4f}"
            )

        return history

    def predict(self, X):
        probabilities = self.forward(X)
        return np.argmax(probabilities, axis=1)

    def accuracy(self, X, y_true):
        predictions = self.predict(X)
        return np.mean(predictions == y_true)