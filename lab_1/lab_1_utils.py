import pickle

import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def train_xor_perceptron(epochs, learning_rate):
    inputs = np.array([[0, 0],
                       [0, 1],
                       [1, 0],
                       [1, 1]])

    outputs = np.array([[0],
                        [1],
                        [1],
                        [0]])

    np.random.seed(42)
    input_hidden_weights = np.random.rand(2, 2)
    hidden_output_weights = np.random.rand(2, 1)

    for epoch in range(epochs):
        hidden_layer_input = np.dot(inputs, input_hidden_weights)
        hidden_layer_output = sigmoid(hidden_layer_input)

        final_layer_input = np.dot(hidden_layer_output, hidden_output_weights)
        final_layer_output = sigmoid(final_layer_input)

        error = outputs - final_layer_output
        final_layer_delta = error * sigmoid_derivative(final_layer_output)

        hidden_layer_error = final_layer_delta.dot(hidden_output_weights.T)
        hidden_layer_delta = hidden_layer_error * sigmoid_derivative(hidden_layer_output)

        hidden_output_weights += hidden_layer_output.T.dot(final_layer_delta) * learning_rate
        input_hidden_weights += inputs.T.dot(hidden_layer_delta) * learning_rate

    return input_hidden_weights, hidden_output_weights, final_layer_output


def predict(inputs, input_hidden_weights, hidden_output_weights):
    hidden_layer_input = np.dot(inputs, input_hidden_weights)
    hidden_layer_output = sigmoid(hidden_layer_input)

    final_layer_input = np.dot(hidden_layer_output, hidden_output_weights)
    final_layer_output = sigmoid(final_layer_input)

    return final_layer_output


def save_model(input_hidden_weights, hidden_output_weights, filename="model.pkl"):
    with open(filename, 'wb') as f:
        pickle.dump((input_hidden_weights, hidden_output_weights), f)


def load_model(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


a, b, c = train_xor_perceptron(100000, 0.1)
save_model(a, b)
