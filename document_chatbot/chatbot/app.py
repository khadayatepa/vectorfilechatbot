import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI

# Load .env and get API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
st.write("DEBUG API Key:", api_key)

print("DEBUG API Key:", os.getenv("OPENAI_API_KEY"))

if not api_key:
    st.error("❌ OPENAI_API_KEY not found. Please check your .env file.")
    st.stop()

# Title
st.title("🤖 Document Q&A Chatbot")

# Upload
uploaded_file = st.file_uploader("📄 Upload a PDF file", type=["pdf"])
if uploaded_file:
    os.makedirs("data", exist_ok=True)
    os.makedirs("vectorstore", exist_ok=True)

    # Save file
    file_path = os.path.join("data", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    st.success(f"✅ {uploaded_file.name} uploaded successfully.")

    # Vector path
    index_name = os.path.splitext(uploaded_file.name)[0]
    vector_path = os.path.join("vectorstore", index_name)

    # Load or Create vectorstore
    if os.path.exists(vector_path):
        db = FAISS.load_local(vector_path, OpenAIEmbeddings(openai_api_key=api_key), allow_dangerous_deserialization=True)
        st.info(f"🔄 Loaded existing vector index for '{uploaded_file.name}'")
    else:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        db = FAISS.from_documents(chunks, embeddings)
        db.save_local(vector_path)
        st.success(f"🧠 Vector index created and saved for '{uploaded_file.name}'")

    # Question Input
    query = st.text_input("💬 Ask a question about the document")
    if query:
        with st.spinner("🧠 Thinking..."):
            matched_docs = db.similarity_search(query)
            llm = ChatOpenAI(temperature=0, openai_api_key=api_key)
            chain = load_qa_chain(llm, chain_type="stuff")
            answer = chain.run(input_documents=matched_docs, question=query)
            st.success("💡 Answer:")
            st.write(answer)
