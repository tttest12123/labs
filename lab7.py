import streamlit as st
from lab_7 import code


def show():
    st.title("Lab 7")

    st.header("LSTM for recognizing sensitive text coloring, Yelp Dataset")
    st.text("Unreal to get it up here, and quantisized version is too stupid to show it, so just report")
    st.divider()

    st.text("First we will load our dataset and preprocess it")
    st.code(code.text_preprocessing, language="python")
    st.divider()

    st.text("Then we will tokenize it and also encode")
    st.code(code.tokenization, language="python")
    st.divider()

    st.text("Next step is model creation")
    st.code(code.model, language="python")
    st.divider()

    st.text("Now lets eval our model")
    st.code(code.eval, language="python")
    st.image('lab_7/scor.png')
    st.divider()

    st.text("And finally test it on some phrases")
    st.code(code.predict, language="python")
    st.image('lab_7/scr.png')



