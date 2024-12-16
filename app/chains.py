import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

load_dotenv()
api_key = st.secrets["GROQ_API_KEY"]

class Chain:
    def __init__(self):
        self.llm = ChatGroq(temperature=0, groq_api_key=api_key, model_name="llama-3.3-70b-versatile")

    def process_query(self, data, query):
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            You are an assistant tasked with providing relevant information from the above page content based on the following query: {user_query}.
            ### (NO PREAMBLE):
            """
        )
        query_chain = prompt_extract | self.llm
        res = query_chain.invoke(input={"page_data": data, "user_query": query})
        
        return res.content


if __name__ == "__main__":
    print(os.getenv("GROQ_API_KEY"))