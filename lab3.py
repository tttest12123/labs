import streamlit
import streamlit as st
import tensorflow as tf
from keras.src.datasets import mnist
from matplotlib import pyplot as plt
from tensorflow import keras
from PIL import Image
import numpy as np
from lab_3.show_code import CODE


def preprocess_image(image):
    image = image.resize((28, 28))
    image = image.convert("L")
    image_array = np.array(image)
    image_array = 255 - image_array
    image_array = image_array / 255.0
    image_array = image_array.reshape((1, 28, 28))
    return image_array


def show():
    st.title("Lab 3")

    st.write("MNIST Digit Prediction")
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    model = keras.models.load_model("lab_3/mnist_model.h5")
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1) / 255.0
    predictions = model.predict(X_test)
    predicted_classes = tf.argmax(predictions, axis=1).numpy()

    fig, axs = plt.subplots(3, 5, figsize=(15, 6))
    for i, ax in enumerate(axs.flat):
        ax.imshow(X_test[i].reshape(28, 28), cmap='gray')
        ax.set_title(f'Predicted: {predicted_classes[i]}, Actual: {y_test[i]}')
        ax.axis('off')

    plt.tight_layout()
    st.pyplot(plt)

    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
    st.info('Choose an image, preferrably from mnist dataset', icon="ℹ️")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        preprocessed_image = preprocess_image(image)
        predictions = model.predict(preprocessed_image)
        predicted_class = np.argmax(predictions)

        st.write(f"Predicted Digit: {predicted_class}")
    st.write("## Training Code")
    st.code(CODE, language='python')
