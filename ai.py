import requests
from llama_cpp import Llama


def generate_text_from_prompt(user_prompt,
                              max_tokens=100,
                              temperature=0.3,
                              top_p=0.1,
                              echo=True,
                              stop=["Q", "\n"]):
    import streamlit as st
    import os

    file_url = "https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF/resolve/main/zephyr-7b-beta.Q4_0.gguf?ref=localhost"
    st.markdown(f"Download the file from [this link]({file_url})")

    # Ensure the .model directory exists
    os.makedirs('.model', exist_ok=True)

    # Button to trigger file download
    if st.button("Download File"):
        response = requests.get(file_url)
        if response.status_code == 200:
            file_path = os.path.join('.model', 'zephyr-7b-beta.Q4_0.gguf')
            with open(file_path, "wb") as f:
                f.write(response.content)
            st.success(f"File downloaded successfully to {file_path}!")
        else:
            st.error("Failed to download file.")

    my_model_path = "./model/zephyr-7b-beta.Q4_0.gguf"
    CONTEXT_SIZE = 512
    if st.button("a File"):
        zephyr_model = Llama(model_path=my_model_path,
                             n_ctx=CONTEXT_SIZE)

        model_output = zephyr_model(
            user_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            echo=echo,
            stop=stop,
        )

        return model_output
