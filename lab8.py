import streamlit as st
from lab_8 import code


def show():
    st.title("Lab 8")

    st.header("СNN-bi-LSTM for speech-to-text, LJ-Speech")
    st.text("Unreal to get it up here, and quantisized version is too stupid to show it, so just report")
    st.divider()
    arr = [
        'it is of the first importance that the letter used should be fine in form;',
        'especially as no more time is occupied, or cost incurred, in casting, setting, or printing beautiful letters',
        'than in the same operations with ugly ones.',
        'And it was a matter of course that in the Middle Ages, when the craftsmen took care that beautiful form should always be a part of their productions whatever they were,'
    ]
    arr2 = [
        'it is of utmost importance that the letters use should be fine in form',
        'specially as no more time is occupied or cost incurred in casting setting or printing beautiful letters',
        'than in the same operations with unattractive ones',
        'and it was naturally expected that in the midle ages craftsmen ensure that beautiful form was always part of their productions, whatever they were'
    ]

    aud = 'lab_8/LJ001-001'
    st.write('Examples of audio processing')
    for i in range(1, 5):
        st.audio(f'{aud}2.wav')
        st.write(f"Expected: {arr[i-1]}")
        st.write(f"Got: {arr2[i-1]}")

    st.divider()


    st.text("Model training code")
    st.code(code.model_creation, language="python")
    st.divider()

    st.text("Evaluating and testing")
    st.code(code.predict, language="python")
    st.divider()







