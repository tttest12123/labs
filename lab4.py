import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np


from lab_4.lab_4_utils import imagenette_classes
from lab_4.show_code import CODE


def load_model():
    return tf.keras.models.load_model('lab_4/alexnet.h5')

def predict(model, img, class_names):
    img = tf.image.resize(img, (224, 224))
    img = tf.cast(img, tf.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    predictions = model.predict(img)
    predicted_index = np.argmax(predictions, axis=1)[0]
    predicted_label = class_names[predicted_index]
    return predicted_label

def show():
    model = load_model()
    print(imagenette_classes)
    st.title('Imagenette Classifier')

    st.write('## Upload an Image for Prediction')
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = image.load_img(uploaded_file, target_size=(224, 224))
        img_array = image.img_to_array(img)
        st.image(img_array / 255.0, caption='Uploaded Image', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        pred_label = predict(model, img_array, imagenette_classes)
        st.write(f'Predicted Label: {pred_label}')


    st.write("## Training Code")
    st.code(CODE, language='python')