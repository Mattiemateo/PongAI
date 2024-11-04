import numpy as np
import os

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, weights_file='pong_weights.npz'):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.weights_file = weights_file
        
        if os.path.exists(weights_file):
            self.load_weights()
        else:
            self.weights1 = np.random.randn(input_size, hidden_size) / np.sqrt(input_size)
            self.weights2 = np.random.randn(hidden_size, output_size) / np.sqrt(hidden_size)
        
        print(f"Weights shapes: {self.weights1.shape}, {self.weights2.shape}")

    def save_weights(self):
        np.savez(self.weights_file, weights1=self.weights1, weights2=self.weights2)
        print("Weights saved successfully.")

    def load_weights(self):
        if os.path.exists(self.weights_file):
            data = np.load(self.weights_file)
            self.weights1 = data['weights1']
            self.weights2 = data['weights2']
            print("Weights loaded successfully.")
        else:
            print("Weights file not found. Using random initialization.")


    def feedforward(self, X):
        self.hidden = sigmoid(np.dot(X, self.weights1))
        self.output = sigmoid(np.dot(self.hidden, self.weights2))
        return self.output

    def train(self, X, y, learning_rate, epochs):
        for _ in range(epochs):
            # Feedforward
            hidden = sigmoid(np.dot(X, self.weights1))
            output = sigmoid(np.dot(hidden, self.weights2))
            
            # Backpropagation
            output_error = y - output
            output_delta = output_error * sigmoid_derivative(output)
            
            hidden_error = np.dot(output_delta, self.weights2.T)
            hidden_delta = hidden_error * sigmoid_derivative(hidden)
            
            # Update weights
            self.weights2 += learning_rate * np.dot(hidden.T, output_delta)
            self.weights1 += learning_rate * np.dot(X.T, hidden_delta)
        
        self.save_weights()
        print(f"Training complete. Final error: {np.mean(np.abs(y - self.feedforward(X)))}")

    def get_weights(self):
        return [self.weights1, self.weights2]

    def set_weights(self, weights):
        self.weights1, self.weights2 = weights
        self.save_weights()
