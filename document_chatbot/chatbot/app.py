
import streamlit as st
from loader import load_and_split
from vectorstore import create_vectorstore
from chatbot import get_conversational_chain
import os
from dotenv import load_dotenv

load_dotenv()

st.title("📚 Document Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file:
    with open(f"data/{uploaded_file.name}", "wb") as f:
        f.write(uploaded_file.read())
    
    chunks = load_and_split(f"data/{uploaded_file.name}")
    create_vectorstore(chunks)
    st.success("Document loaded and indexed!")

query = st.text_input("Ask a question about the document:")

if query:
    chain = get_conversational_chain()
    result = chain({"question": query, "chat_history": st.session_state.chat_history})
    
    st.session_state.chat_history.append((query, result["answer"]))
    
    for q, a in st.session_state.chat_history:
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Bot:** {a}")
