import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI

# streamlit run chatbot.py 

# GOOGLE_API_KEY
import os
assert os.environ.get("GOOGLE_API_KEY")

st.set_page_config(page_title="🦜🔗 Chatbot")
st.title("🦜🔗 Chatbot")

def generate_response(input_text):
    llm = ChatGoogleGenerativeAI(model="gemini-pro",
                                temperature=0)
    st.info(llm.invoke(input_text).content)

with st.form("Question"):
    # 첫 페이지가 실행될 때 보여줄 질문
    text = st.text_area('질문 입력:', 'What types of text models does OpenAI provide?')
    submitted = st.form_submit_button("보내기")
    generate_response(text)