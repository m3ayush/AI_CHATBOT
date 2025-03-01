import os
import streamlit as st
from dotenv import load_dotenv

from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq

load_dotenv()
API_KEY = os.getenv("API_KEY")

def initialize_llm():
    llm = ChatGroq(
        temperature=0.7,
        groq_api_key=API_KEY,
        model_name="llama-3.3-70b-versatile"
    )
    return llm

def create_vector_db():
    loader = DirectoryLoader("./data/", glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    embeddings = HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vector_db = Chroma.from_documents(texts, embeddings, persist_directory='./chroma_db')
    vector_db.persist()

    st.success("ChromaDB created and data saved")
    return vector_db

def setup_qa_chain(vector_db, llm):
    retriever = vector_db.as_retriever()
    prompt_templates = """You are a compassionate mental health chatbot. Respond thoughtfully to the following question:
    {context}
    User: {question}
    Chatbot: """
    PROMPT = PromptTemplate(template=prompt_templates, input_variables=['context', 'question'])
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain

def main():
    st.title("Mental Health Chatbot")
    st.write("Welcome to the Mental Health Chatbot. How can I assist you today?")

    llm = initialize_llm()

    db_path = "./chroma_db/"
    if not os.path.exists(db_path):
        st.write("Creating ChromaDB...")
        vector_db = create_vector_db()
    else:
        embeddings = HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        vector_db = Chroma(persist_directory=db_path, embedding_function=embeddings)

    qa_chain = setup_qa_chain(vector_db, llm)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    user_input = st.chat_input("Type your message here...")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.write(user_input)

        with st.spinner("Thinking..."):
            response = qa_chain.run(user_input)

        response = qa_chain.run(user_input)

        st.session_state.chat_history.append({"role": "assistant", "content": response})

        with st.chat_message("assistant"):
            st.write(response)

if __name__ == "__main__":
    main()