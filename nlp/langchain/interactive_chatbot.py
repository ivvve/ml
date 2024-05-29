import streamlit as st
from streamlit_chat import message
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.document_loaders import PyPDFLoader
import tempfile

# streamlit run interactive_chatbot.py 

import os
assert os.environ.get("GOOGLE_API_KEY")

# PDF 파일 업로드
uploaded_file = st.sidebar.file_uploader("upload", type="pdf")
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    
    loader = PyPDFLoader(tmp_file_path)
    data = loader.load()

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_documents(data, embeddings)

    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatGoogleGenerativeAI(model="gemini-pro", temperature=0),
        retriever=vector_store.as_retriever(),
        verbose=True,
    )

    def conversational_chat(query):
        # https://python.langchain.com/v0.1/docs/use_cases/question_answering/chat_history/#chain-with-chat-history
        result = chain.invoke({
            "question": query,
            "chat_history": st.session_state["history"],
        })
        # 세션에 대화 내역 추가
        st.session_state["history"].append((query, result["answer"]))
        return result["answer"]

    print("[session state]")
    for key, val in st.session_state.items():
        print(f"{key}: {val}")

    if "history" not in st.session_state:
        st.session_state["history"] = []

    if "generated" not in st.session_state:
        st.session_state["generated"] = [f"안녕하세요! {uploaded_file.name}에 관해 질문 주세요."]

    if "past" not in st.session_state:
        st.session_state["past"] = ["안녕하세요!"]

    # 챗봇 대화 내역 컨테이너 UI
    response_container = st.container()
    container = st.container()
    with container:
        with st.form(key="Conv_Question", clear_on_submit=True):
            user_input = st.text_input("Query:", placeholder="PDF 파일에 대해 얘기해볼까요?", key="input")
            submit_button = st.form_submit_button(label="Send")

        # 입력창에 입력을 하고 버튼 클릭
        if submit_button and user_input:
            output = conversational_chat(user_input)
            st.session_state["past"].append(user_input)
            st.session_state["generated"].append(output)

    if st.session_state["generated"]:
        with response_container:
            for i in range(len(st.session_state["generated"])):
                message(st.session_state["past"][i], is_user=True, key=f"{i}_user", avatar_style="fun-emoji", seed="Nala")
                message(st.session_state["generated"][i], key=str(i), avatar_style="bottts", seed="Fluffy")
