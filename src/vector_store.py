from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
print(OPENAI_API_KEY)


def create_vector_store(text_data):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    texts = []
    for paper in text_data:
        chunks = text_splitter.split_text(paper["text"])
        texts.extend([(chunk, paper["filename"]) for chunk in chunks])

    doc_texts, metadata = zip(*texts)
    vector_store = FAISS.from_texts(doc_texts, embeddings, metadatas=[{"source": m} for m in metadata])
    return vector_store

def save_vector_store(vector_store, path="vector_store/"):
    vector_store.save_local(path)

def load_vector_store(path="vector_store/"):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    return FAISS.load_local(path, embeddings)
