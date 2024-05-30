import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import PromptTemplate

# streamlit run mail_writer.py 

def get_email():
    input_text = st.text_area(
        label="메일 입력", label_visibility="collapsed",
        placeholder="당신의 메일은...", key="input_text",
    )
    return input_text

def load_llm():
    import os
    assert os.environ.get("GOOGLE_API_KEY")

    llm = ChatGoogleGenerativeAI(model="gemini-pro",
                                temperature=0)
    return llm

query_template = """
    메일을 작성해주세요.
    아래는 이메일입니다.
    이메일: {email}
"""

prompt = PromptTemplate(
    input_variables=["email"],
    template=query_template,
)

st.set_page_config(page_title="이메일 작성 서비스", page_icon=":robot:")
st.header("이메일 작성기")
input_text = get_email()

st.button("*예제를 보여주세요*", type="secondary", help="봇이 작성한 메일을 확인해보세요.")
st.markdown("### 봇이 작성한 메일은:")

if input_text:
    llm = load_llm()

    prompt_with_email = prompt.format(email=input_text)
    formatted_email = llm.predict(prompt_with_email)

    st.write(formatted_email)