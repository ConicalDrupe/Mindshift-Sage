import streamlit as st
from sage_model import retrieve_insight

st.set_page_config(page_title='Mindshift Sage',
                   page_icon=':herb:',
                   layout="centered",
                   initial_sidebar_state="expanded",
                   menu_items={
                       'Get Help': 'https://data-boon.com',
                       'About': "# This is a header. Wisdom with a *click* of a button"})

st.title('Mindshift Sage :herb:')


def chat(i):
    if i>0:
        user_text = st.text_input("Anything else on your mind?",key=str(i))
    else:
        user_text = st.text_input("What is going on in your life?",key=str(i))

    if user_text:
        response = retrieve_insight(user_text)
        st.write(response)
        chat(i+1)

with st.chat_message("Sage AI"):
    chat(0)
