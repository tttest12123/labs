import tensorflow as tf
import streamlit as st
import cv2
import numpy as np

from lab_6.show_code import CODE
from huggingface_hub import hf_hub_download


def load_model_from_hf(repo_id, filename):
    model_path = hf_hub_download(repo_id=repo_id, filename=filename)
    return tf.keras.models.load_model(model_path)


def show():
    model_path = 'rriaa/lab_6'
    model_name = 'xception.h5'
    model = load_model_from_hf(model_path, model_name)
    time = st.number_input("Input number of secods to process: ", 1, max_value=5)
    st.title("Lab 6")
    uploaded_video = "vid.mp4"

    if 'timestamps' not in st.session_state:
        st.session_state.timestamps = []

    if uploaded_video is not None:
        video = cv2.VideoCapture(uploaded_video)
        stframe = st.empty()
        fps = video.get(cv2.CAP_PROP_FPS)
        max_frames = int(time * fps)

        frame_count = 0

        while video.isOpened() and frame_count < max_frames:
            ret, frame = video.read()
            if not ret:
                break

            input_frame = cv2.resize(frame, (299, 299))
            input_frame = np.expand_dims(input_frame, axis=0)
            input_frame = input_frame / 255.0
            predictions = model.predict(input_frame)
            if np.max(predictions) > 0:
                timestamp = video.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                st.session_state.timestamps.append(round(timestamp, 2))

            stframe.image(frame, channels="BGR")
            frame_count += 1

        video.release()

        with st.container():
            st.write("Timestamps with logo detected:")
            for i in range(0, len(st.session_state.timestamps), 5):
                cols = st.columns(5)
                for j, col in enumerate(cols):
                    if i + j < len(st.session_state.timestamps):
                        col.write(f"{st.session_state.timestamps[i + j]:.2f} seconds")
    else:
        st.write("Please upload a video file")

    st.write("## Training Code")
    st.code(CODE, language='python')
