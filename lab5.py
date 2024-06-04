import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from huggingface_hub import hf_hub_download

class_names = [
    "Chihuahua", "Japanese Spaniel", "Maltese Dog", "Pekinese", "Shih-Tzu", "Blenheim Spaniel",
    "Papillon", "Toy Terrier", "Rhodesian Ridgeback", "Afghan Hound", "Basset Hound", "Beagle",
    "Bloodhound", "Bluetick", "Black-and-tan Coonhound", "Walker Hound", "English Foxhound",
    "Redbone", "Borzoi", "Irish Wolfhound", "Italian Greyhound", "Whippet", "Ibizan Hound",
    "Norwegian Elkhound", "Otterhound", "Saluki", "Scottish Deerhound", "Weimaraner",
    "Staffordshire Bullterrier", "American Staffordshire Terrier", "Bedlington Terrier",
    "Border Terrier", "Kerry Blue Terrier", "Irish Terrier", "Norfolk Terrier",
    "Norwich Terrier", "Yorkshire Terrier", "Wire-Haired Fox Terrier", "Lakeland Terrier",
    "Sealyham Terrier", "Airedale", "Cairn", "Australian Terrier", "Dandie Dinmont",
    "Boston Bull", "Miniature Schnauzer", "Giant Schnauzer", "Standard Schnauzer",
    "Scottish Terrier", "Tibetan Terrier", "Silky Terrier", "Soft-Coated Wheaten Terrier",
    "West Highland White Terrier", "Lhasa", "Flat-Coated Retriever", "Curly-Coated Retriever",
    "Golden Retriever", "Labrador Retriever", "Chesapeake Bay Retriever", "German Short-Haired Pointer",
    "Vizsla", "English Setter", "Irish Setter", "Gordon Setter", "Brittany Spaniel",
    "Clumber", "English Springer", "Welsh Springer Spaniel", "Cocker Spaniel", "Sussex Spaniel",
    "Irish Water Spaniel", "Kuvasz", "Schipperke", "Groenendael", "Malinois", "Briard",
    "Kelpie", "Komondor", "Old English Sheepdog", "Shetland Sheepdog", "Collie", "Border Collie",
    "Bouvier Des Flandres", "Rottweiler", "German Shepherd", "Doberman", "Miniature Pinscher",
    "Greater Swiss Mountain Dog", "Bernese Mountain Dog", "Appenzeller", "Entlebucher",
    "Boxer", "Bull Mastiff", "Tibetan Mastiff", "French Bulldog", "Great Dane", "Saint Bernard",
    "Eskimo Dog", "Malamute", "Siberian Husky", "Affenpinscher", "Basenji", "Pug",
    "Leonberg", "Newfoundland", "Great Pyrenees", "Samoyed", "Pomeranian", "Chow",
    "Keeshond", "Brabancon Griffon", "Pembroke", "Cardigan", "Toy Poodle", "Miniature Poodle",
    "Standard Poodle", "Mexican Hairless", "Dingo", "Dhole", "African Hunting Dog"
]

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
    st.title('Dog Breed Classifier')

    st.write('## Upload an Image for Prediction')
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = image.load_img(uploaded_file, target_size=(299, 299))
        img_array = image.img_to_array(img)
        st.image(img_array / 255.0, caption='Uploaded Image', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        pred_label = predict(model, img_array, class_names)
        st.write(f'Predicted Label: {pred_label}')

