import streamlit as st

def show():
    video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    if video_url:
        #comit
        col1, col2, col3 = st.columns(3)
        for col in (col1, col2, col3):
            with col:
                for i in range(5):
                    st.video(video_url, autoplay=True, muted=True)

