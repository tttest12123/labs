import streamlit as st
import lab1
import lab2
import lab3
import lab4
import lab5
import lab6
import lab7
import lab8

pages = {
    "Lab 1": lab1.show,
    "Lab 2": lab2.show,
    "Lab 3": lab3.show,
    "Lab 4": lab4.show,
    "Lab 5": lab5.show,
    "Lab 6": lab6.show,
    "Lab 7": lab7.show,
    "Lab 8": lab8.show,
}

st.sidebar.title("Labs")
page = st.sidebar.radio("Go to", list(pages.keys()))

pages[page]()
