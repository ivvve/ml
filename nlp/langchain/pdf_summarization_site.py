from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def process_text(text):
    # text를 chunk로 분할
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1_000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)

    # 벡터 변환
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    documents = FAISS.from_texts(chunks, embeddings)
    return documents

import streamlit as st
from PyPDF2 import PdfReader # pip install PyPDF2
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain

import os

def main():
    assert os.environ.get("GOOGLE_API_KEY")

    st.title("📝 PDF 요약")
    st.divider()

    pdf = st.file_uploader("PDF 파일을 업로드해주세요", type="pdf")

    if not pdf:
        return

    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    
    # chunk로 분할하여 vector화된 데이터
    documents = process_text(text)
    query = "업로드된 PDF 파일의 내용을 약 3~5 문장으로 요약해주세요."
    docs = documents.similarity_search(query)

    llm = ChatGoogleGenerativeAI(model="gemini-pro",
                                 temperature=0.1)
    chain = load_qa_chain(llm, chain_type="stuff", verbose=True)
    response = chain.run(input_documents=docs, question=query)

    st.subheader("-- 요약 결과 --")
    st.write(response)

if __name__ == "__main__":
    main()
