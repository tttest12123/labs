import streamlit as st
import lab1
import lab2
import lab3
import lab4
import lab5
import lab6
import lab7
import lab8
import bonus
from transformers import T5Tokenizer, T5ForConditionalGeneration

pages = {
    "Lab 1": lab1.show,
    "Lab 2": lab2.show,
    "Lab 3": lab3.show,
    "Lab 4": lab4.show,
    "Lab 5": lab5.show,
    "Lab 6": lab6.show,
    "Lab 7": lab7.show,
    "Lab 8": lab8.show,
    "Bonus": bonus.show,
}

st.sidebar.title("Labs")
page = st.sidebar.radio("Go to", list(pages.keys()))

pages[page]()
st.markdown("---")
st.info('Ask AI about this app!', icon='💻')

@st.cache_resource
def load_model():
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
    return tokenizer, model

tokenizer, model = load_model()

input_text = st.text_input("Enter your text:")

if st.button("Generate"):
    if input_text:
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids
        outputs = model.generate(input_ids)
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.write(f"Output: {decoded_output}")
    else:
        st.write("Please enter some text to process.")

st.warning("Yes, it's stupid. The Streamlit community gives 1 GB of space, and not a lot of models can fit in it. "
           "It's better to use an API if you will be trying to recreate it yourself, but mind that "
           "I have an evaluation in the second lab, so no secure tokens for you)")
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("Created by Mariia Kovalenko")

with col2:
    st.markdown(":frog:")

with col3:
    st.markdown("[LinkedIn](https://www.linkedin.com/in/mariia-k-bbbb75266/)")
