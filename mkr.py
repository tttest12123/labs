import streamlit as st
import lab_1.lab_1_utils as l
import numpy as np

from lab_1.show_code import CODE
from mkr_utils.code import yolo


def show():
    st.title("Mkr")

    st.header("YOLO 8 on Mcdonald logo datasets")
    st.text("Making full logic here was too resource consuming, so just interactive video elements.")

    st.text("Creating augumented dataset:")
    st.image('mkr_utils/train_batch0.jpg', caption='Training Batch 0', use_column_width=True)

    st.text("Training yolo8:")

    st.code(yolo, language="python")
    st.text("Predicted labels on validation set")
    st.image('mkr_utils/val_batch0_pred.jpg', caption='Validation Batch 0 Predictions', use_column_width=True)
    col1, col2 = st.columns(2)
    with col1:
        st.text("Confusion_matrix")
        st.image('mkr_utils/confusion_matrix.png', caption='Confusion Matrix', use_column_width=True)
    with col2:
        st.text("F1_curve")
        st.image('mkr_utils/F1_curve.png', caption='F1 Curve', use_column_width=True)
    st.text("Training results")
    st.image('mkr_utils/results.png', caption='Results', use_column_width=True)

    st.title("Outputs")

    st.image('mkr_utils/scrshot.png', caption='Screenshot', use_column_width=True)
    col1, col2 = st.columns(2)
    with col1:
        st.video('mkr_utils/test1.mp4' )

    with col2:
        st.video('mkr_utils/out.mp4', format='video/avi')
        st.link_button(label="View video (in case of fail)",
                   url="https://youtube.com/shorts/1oyANrMCwE4?feature=share")
