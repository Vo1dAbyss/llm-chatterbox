import streamlit as st
from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv
import random

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

## PAGE SETTINGS ##
st.title("LLM Chat")

## CLIENT SETTINGS ##
model_name = "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO"

# Generate the response
def generate_response():
    # Generate the response (stream)
    for token in st.session_state.client.chat_completion(
        messages=st.session_state.messages, 
        stream=True, 
        frequency_penalty=0.7, 
        temperature=0.7,
        max_tokens=1000, 
        seed=random.randint(0, 2**64)
    ):
        yield token.choices[0].delta.content

def main():
    
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = "You are a human, and have completelly human like emotions. You talk as in a roleplay."

    # Check if the messages / client exist in the session, else create them.
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "system", "content": st.session_state.system_prompt}]

    if "client" not in st.session_state:
        st.session_state.client = InferenceClient(model_name, token=HF_TOKEN)

    sidebar = st.sidebar
    system_prompt_input = sidebar.text_input(label="System Prompt", value=st.session_state.system_prompt)

    if system_prompt_input != st.session_state.system_prompt:
        st.session_state.system_prompt = system_prompt_input
        st.session_state.messages = [
            {"role": "system", "content": st.session_state.system_prompt}
        ]
    
    # Add all the messages
    for message in st.session_state.messages:
        if message["role"] != "system":
            with st.chat_message(message["role"]):
                st.write(message["content"])

    # Response process
    if chat_input := st.chat_input("What are you up to?"):
        with st.chat_message("user"):
            st.write(chat_input)   
        st.session_state.messages.append({"role": "user", "content": chat_input})

        response = ""
        with st.chat_message("assistant"):
            token = st.write_stream(generate_response())
            response += token
        st.session_state.messages.append({"role": "assistant", "content": response})
        

if __name__ == "__main__":
    main()