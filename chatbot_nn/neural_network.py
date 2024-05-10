import numpy as np


def sigmoid(x):
    """The sigmoid activation function."""
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    """Derivative of the sigmoid function."""
    return x * (1 - x)


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases
        self.w1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.w2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))

    def feedforward(self, X):
        # Layer 1
        self.z1 = np.dot(X, self.w1) + self.b1
        self.a1 = sigmoid(self.z1)

        # Layer 2 (output layer)
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = sigmoid(self.z2)

        return self.a2

    def backpropagation(self, X, y):
        # Forward pass
        output = self.feedforward(X)

        # Backward pass (error and gradients calculation)
        error = output - y
        d_w2 = np.dot(self.a1.T, error * sigmoid_derivative(output))
        d_b2 = np.sum(error * sigmoid_derivative(output), axis=0)

        error_hidden = np.dot(error, self.w2.T)
        d_w1 = np.dot(X.T, error_hidden * sigmoid_derivative(self.a1))
        d_b1 = np.sum(error_hidden * sigmoid_derivative(self.a1), axis=0)

        # Update weights and biases
        learning_rate = 0.1
        self.w1 -= learning_rate * d_w1
        self.b1 -= learning_rate * d_b1
        self.w2 -= learning_rate * d_w2
        self.b2 -= learning_rate * d_b2

    def train(self, X, y, epochs=1000):
        for epoch in range(epochs):
            self.backpropagation(X, y)
            if epoch % 100 == 0:
                loss = np.mean((y - self.feedforward(X)) ** 2)
                print(f'Epoch {epoch}, Loss: {loss:.4f}')
