import streamlit as st
import lab_1.lab_1_utils as l
import numpy as np

from lab_1.show_code import CODE


def show():
    st.title("Lab 1")

    st.header("XOR Perceptron Training Demo")
    st.write("Train a simple neural network to solve the XOR problem.")

    epochs = 10000
    learning_rate = 0.01

    if 'input_hidden_weights' not in st.session_state or 'hidden_output_weights' not in st.session_state:
        input_hidden_weights, hidden_output_weights = l.load_model("lab_1/model.pkl")
        st.session_state.input_hidden_weights = input_hidden_weights
        st.session_state.hidden_output_weights = hidden_output_weights
    else:
        input_hidden_weights = st.session_state.input_hidden_weights
        hidden_output_weights = st.session_state.hidden_output_weights

    st.write("## Test the Trained Perceptron")

    input1 = st.number_input("Input 1", min_value=0, max_value=1, step=1, value=0)
    input2 = st.number_input("Input 2", min_value=0, max_value=1, step=1, value=0)

    if st.button("Test Perceptron"):
        user_input = np.array([[input1, input2]])
        prediction = l.predict(user_input, input_hidden_weights, hidden_output_weights)
        st.write("### Prediction")
        st.write(f"Input: [{input1}, {input2}]")
        st.write(f"Output: {prediction[0][0]:.0f}")

    st.write("## Training Code")
    st.code(CODE, language='python')


if __name__ == '__main__':
    show()
