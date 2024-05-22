import streamlit as st
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


def show():
    st.title("Lab 1")

    st.header("XOR Perceptron Training Demo")
    st.write("Train a simple neural network to solve the XOR problem.")

    epochs = 100000
    learning_rate = 0.01

    if 'input_hidden_weights' not in st.session_state or 'hidden_output_weights' not in st.session_state:
        input_hidden_weights, hidden_output_weights, final_layer_output = train_xor_perceptron(epochs, learning_rate)
        st.session_state.input_hidden_weights = input_hidden_weights
        st.session_state.hidden_output_weights = hidden_output_weights
    else:
        input_hidden_weights = st.session_state.input_hidden_weights
        hidden_output_weights = st.session_state.hidden_output_weights

    st.write("## Test the Trained Perceptron")

    input1 = st.number_input("Input 1", min_value=0.0, max_value=1.0, step=1.0, value=0.0)
    input2 = st.number_input("Input 2", min_value=0.0, max_value=1.0, step=1.0, value=0.0)

    if st.button("Test Perceptron"):
        user_input = np.array([[input1, input2]])
        prediction = predict(user_input, input_hidden_weights, hidden_output_weights)
        st.write("### Prediction")
        st.write(f"Input: [{input1}, {input2}]")
        st.write(f"Output: {prediction[0][0]:.0f}")

    # Display the training code at the bottom
    st.write("## Training Code")
    code = """
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
    """
    st.code(code, language='python')


if __name__ == '__main__':
    show()
