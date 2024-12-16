import streamlit as st
from langchain_community.document_loaders import WebBaseLoader

from chains import Chain
from utils import clean_text


def create_streamlit_app(llm, clean_text):
    st.title("🌍 Chat With Any Website")
    url_input = st.text_input("Enter a URL:", value="https://en.wikipedia.org/wiki/Document_AI")
    query = st.text_input("Enter your query:")
    submit_button = st.button("Submit")

    if submit_button:
        try:
            loader = WebBaseLoader([url_input])
            data = clean_text(loader.load().pop().page_content)
            res = llm.process_query(data, query)
            st.write(res)
            
        except Exception as e:
            st.error(f"An Error Occurred: {e}")


if __name__ == "__main__":
    chain = Chain()
    st.set_page_config(layout="wide", page_title="Chat with Website", page_icon="📧")
    create_streamlit_app(chain, clean_text)


