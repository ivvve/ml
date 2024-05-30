# !pip install unstructured
# !pip install sentence-transformers
# !pip install chromadb
# !pip install openai

# Text load and split in chunks
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_docs(documents, chunk_size=1_000, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, 
                                                   chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs

documents = TextLoader("./data/AI.txt").load()
docs = split_docs(documents)

# vector db에 vectorized document 저장
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma.from_documents(docs, embeddings)

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain

import os
assert os.environ.get("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-pro",
                            temperature=0)
chain = load_qa_chain(llm, chain_type="stuff", verbose=True)

query = "AI란?"
matching_docs = db.similarity_search(query)
answer = chain.run(input_documents=matching_docs, question=query)
print(answer)
