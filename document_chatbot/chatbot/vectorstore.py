
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import os

def create_vectorstore(chunks):
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local("vectorstore")
    return db

def load_vectorstore():
    embeddings = OpenAIEmbeddings()
    db = FAISS.load_local("vectorstore", embeddings)
    return db
