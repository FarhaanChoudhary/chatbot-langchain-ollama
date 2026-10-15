
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama

import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()


langchain_key = os.getenv("LANGCHAIN_API_KEY")

if not langchain_key:
    raise ValueError("LangSmith API key is missing")

os.environ["LANGCHAIN_TRACING_V2"] = "true"
## Prompt Template

prompt=ChatPromptTemplate.from_messages(
    [
    ("system", "You are a helpful assistant.Please respond to the user queries."),
    ("user", "Questions:{questions}"),
    ]
)

##streamlit framework
st.title("LangChain Ollama Chatbot")
input_text=st.text_input("Search the topic you want to know about")

##Ollama Llama2 LLM
llm = Ollama(
    model="llama3.2:1b",
    num_ctx=2048
)
output_parser=StrOutputParser() 
chain=prompt | llm | output_parser

if input_text:
    st.write(chain.invoke({"questions":input_text}))