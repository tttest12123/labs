import streamlit as st
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import os

def list_files(starting_directory):
    for root, dirs, files in os.walk(starting_directory):
        st.write(f"Directory: {root}")
        for file in files:
            st.write(f"  {file}")


def train_network(X_train, X_test, y_train, y_test, network_type, layer_config):
    if network_type == 'feed_forward_backprop':
        hidden_layer_sizes = layer_config
        nn = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, max_iter=1000)
    elif network_type == 'cascade_forward_backprop':
        hidden_layer_sizes = layer_config
        nn = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation='logistic', solver='lbfgs', max_iter=1000)
    elif network_type == 'elman_backprop':
        hidden_layer_sizes = layer_config
        nn = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation='tanh', solver='adam', max_iter=1000)

    nn.fit(X_train, y_train)
    y_pred_train = nn.predict(X_train)
    y_pred_test = nn.predict(X_test)

    r2_train = nn.score(X_train, y_train)
    r2_test = nn.score(X_test, y_test)

    mse_train = np.mean((y_pred_train - y_train) ** 2)
    mse_test = np.mean((y_pred_test - y_test) ** 2)

    return r2_train, r2_test, mse_train, mse_test, y_pred_test


def show():
    st.title("Lab 2")

    func_str = st.text_input("Enter a function of two variables (x, y):", "x**2 + y**2")
    st.info('Write as python function, for example sin(x) you will need to write as np.sin(x)', icon="ℹ️")

    network_type = st.selectbox("Select network type",
                                ["feed_forward_backprop", "cascade_forward_backprop", "elman_backprop"])

    if network_type == "feed_forward_backprop":
        layer_config = st.selectbox("Select layer configuration", [(10,), (20,)])
    elif network_type == "cascade_forward_backprop":
        layer_config = st.selectbox("Select layer configuration", [(20,), (10, 10)])
    elif network_type == "elman_backprop":
        layer_config = st.selectbox("Select layer configuration", [(15,), (5, 5, 5)])

    if st.button("Train Network"):
        if "sleep" in func_str:
            st.title("Ігор хватить ламать мені лабу")
            st.image("lab_2/photo_2024-05-23_01-31-59.jpg")
        else:
            def target_func(x, y):
                return eval(func_str)

            X = np.random.uniform(-5, 5, (1000, 2))
            y = target_func(X[:, 0], X[:, 1])
            list_files(os.getcwd())

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            r2_train, r2_test, mse_train, mse_test, y_pred_test = train_network(X_train, X_test, y_train, y_test,
                                                                                network_type, layer_config)
            st.write(f"R^2 (Train): {r2_train:.4f}")
            st.write(f"R^2 (Test): {r2_test:.4f}")
            st.write(f"MSE (Train): {mse_train:.4f}")
            st.write(f"MSE (Test): {mse_test:.4f}")

            x = X_test[:, 0]
            y = X_test[:, 1]
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(x, y, y_pred_test, c='pink', marker='o', label='Predicted')
            ax.scatter(x, y, y_test, c='b', marker='o', label='Real')
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')

            ax.legend()
            st.pyplot(fig)
            st.write("## Training Code")
    code = """
    
    def train_network(X_train, X_test, y_train, y_test, network_type, layer_config):
        if network_type == 'feed_forward_backprop':
            hidden_layer_sizes = layer_config
            nn = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, max_iter=1000)
        elif network_type == 'cascade_forward_backprop':
            hidden_layer_sizes = layer_config
            nn = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation='logistic', solver='lbfgs', max_iter=1000)
        elif network_type == 'elman_backprop':
            hidden_layer_sizes = layer_config
            nn = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation='tanh', solver='adam', max_iter=1000)
    
        nn.fit(X_train, y_train)
    
        y_pred_train = nn.predict(X_train)
        y_pred_test = nn.predict(X_test)
    
        r2_train = nn.score(X_train, y_train)
        r2_test = nn.score(X_test, y_test)
    
        mse_train = np.mean((y_pred_train - y_train) ** 2)
        mse_test = np.mean((y_pred_test - y_test) ** 2)
    
        return r2_train, r2_test, mse_train, mse_test, y_pred_test
   
   
   
    network_type = st.selectbox("Select network type",
                            ["feed_forward_backprop", "cascade_forward_backprop", "elman_backprop"])

    if network_type == "feed_forward_backprop":
        layer_config = st.selectbox("Select layer configuration", [(10,), (20,)])
    elif network_type == "cascade_forward_backprop":
        layer_config = st.selectbox("Select layer configuration", [(20,), (10, 10)])
    elif network_type == "elman_backprop":
        layer_config = st.selectbox("Select layer configuration", [(15,), (5, 5, 5)])
        
    X = np.random.uniform(-5, 5, (1000, 2))
    y = target_func(X[:, 0], X[:, 1])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    r2_train, r2_test, mse_train, mse_test, y_pred_test = train_network(X_train, X_test, y_train, y_test,
                                                                        network_type, layer_config)
    
    
    """
    st.code(code, language="python")



