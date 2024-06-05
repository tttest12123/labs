import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from huggingface_hub import hf_hub_download

from lab_5.show_code import CODE
from lab_5.utils import class_names


def load_model_from_hf(repo_id, filename):
    model_path = hf_hub_download(repo_id=repo_id, filename=filename)
    return tf.keras.models.load_model(model_path)

def predict(model, img, class_names):
    img = tf.image.resize(img, (299, 299))
    img = tf.cast(img, tf.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    predictions = model.predict(img)
    predicted_index = np.argmax(predictions, axis=1)[0]
    predicted_label = class_names[predicted_index]
    return predicted_label

def show():
    model = load_model_from_hf('rriaa/lab_5_model', 'stanford_dogs_inception_v3_model.h5')
    st.title('Lab 5')

    st.write('## Upload an Image for Prediction (dog breed)')
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = image.load_img(uploaded_file, target_size=(299, 299))
        img_array = image.img_to_array(img)
        st.image(img_array / 255.0, caption='Uploaded Image', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        pred_label = predict(model, img_array, class_names)
        st.write(f'Predicted Label: {pred_label}')

    st.write("## Training Code")
    st.code(CODE, language='python')


