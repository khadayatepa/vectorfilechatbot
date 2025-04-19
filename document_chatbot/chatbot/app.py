from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI

# Load environment variables from .env (if present)
load_dotenv()

# Title
st.title("🤖 Document Q&A Chatbot")

# --- 1. API Key Input ---
#api_key = st.text_input("🔑 Enter your OpenAI API key", type="password")
#if not api_key:
#    st.warning("Please enter your OpenAI API key to continue.")
#    st.stop()
#os.environ["OPENAI_API_KEY"] = api_key

# --- 2. PDF Upload ---
uploaded_file = st.file_uploader("📄 Upload a PDF file", type=["pdf"])
if uploaded_file:
    # Ensure the data directory exists
    os.makedirs("data", exist_ok=True)

    # Save uploaded file
    file_path = os.path.join("data", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    st.success(f"✅ {uploaded_file.name} uploaded successfully.")

    # --- 3. Load and Split PDF ---
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    # --- 4. Create Embeddings and Vector Store ---
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    db = FAISS.from_documents(chunks, embeddings)

    # --- 5. User Query ---
    query = st.text_input("💬 Ask a question about the document")
    if query:
        with st.spinner("🧠 Thinking..."):
            matched_docs = db.similarity_search(query)
            llm = ChatOpenAI(temperature=0, openai_api_key=api_key)
            chain = load_qa_chain(llm, chain_type="stuff")
            answer = chain.run(input_documents=matched_docs, question=query)
            st.success("💡 Answer:")
            st.write(answer)
