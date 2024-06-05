import streamlit as st
import lab_1.lab_1_utils as l
import numpy as np

from lab_1.show_code import CODE


def show():
    st.title("Mkr")

    st.header("YOLO 8 on Mcdonalds logo datases")
    st.info('Used roboflow for dataset creation and annotating')


if __name__ == '__main__':
    show()
