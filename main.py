import streamlit as st
import openai
from openai import OpenAI
import lab1
import lab2
import lab3
import lab4
import lab5
import lab6
import lab7
import lab8
import mkr
import bonus

pages = {
    "Lab 1": lab1.show,
    "Lab 2": lab2.show,
    "Lab 3": lab3.show,
    "Lab 4": lab4.show,
    "Lab 5": lab5.show,
    "Lab 6": lab6.show,
    "Lab 7": lab7.show,
    "Lab 8": lab8.show,
    "Mkr": mkr.show,
    "Bonus": bonus.show,
}

st.sidebar.title("Labs")
page = st.sidebar.radio("Go to", list(pages.keys()))

pages[page]()



st.markdown("---")
st.info('Ask AI your questions', icon='💻')


input_text = st.text_input("Enter your text (max 100 characters):", max_chars=100)

if st.button("Generate"):
    if input_text:
        try:
            key = st.secrets['openai_api_key']

            client = OpenAI(api_key=key)

            stream = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": input_text}],
                stream=True,
            )
            output_container = st.empty()
            output = ""
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    output += chunk.choices[0].delta.content
                    output_container.write(output)
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.write("Please enter some text to process.")

st.markdown("---")



col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("Created by Mariia Kovalenko")

with col2:
    st.markdown(":frog:")

with col3:
    st.markdown("[LinkedIn](https://www.linkedin.com/in/mariia-k-bbbb75266/)")
