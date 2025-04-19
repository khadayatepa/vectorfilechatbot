
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from vectorstore import load_vectorstore

def get_conversational_chain():
    vectorstore = load_vectorstore()
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )
    return chain
