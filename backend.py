import os
import streamlit as st
from operator import itemgetter
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Cache embedding model so Streamlit UI doesn't freeze on startup
@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def build_vector_store(pdf_path: str):
    """Loads a PDF resume cleanly using PyMuPDF, chunks the text, and stores vectors in ChromaDB."""
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(documents)
    
    embeddings = get_embedding_model()
    
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    return vector_store

def format_docs(docs):
    """Combines retrieved document chunks into a single string context."""
    return "\n\n".join(doc.page_content for doc in docs)

def get_qa_chain():
    """Initializes and returns the RAG pipeline that outputs both answer and sources."""
    embeddings = get_embedding_model()
    
    vector_store = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is missing! Please check your .env file.")
        
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        groq_api_key=api_key
    )
    
    system_prompt = (
        "You are an AI assistant representing the candidate based on their resume.\n"
        "Answer questions accurately using ONLY the context provided below.\n"
        "If the answer is not contained in the context, state that clearly.\n\n"
        "Context:\n{context}\n\n"
        "Question: {input}"
    )
    
    prompt = ChatPromptTemplate.from_template(system_prompt)
    
    # Modern LCEL parallel chain that returns {"answer": ..., "source_documents": ...}
    retrieval_setup = RunnableParallel(
        context=retriever | format_docs,
        input=RunnablePassthrough(),
        source_documents=retriever
    )
    
    rag_chain = (
        retrieval_setup
        | RunnableParallel(
            answer={
                "context": itemgetter("context"),
                "input": itemgetter("input")
            } | prompt | llm | StrOutputParser(),
            source_documents=itemgetter("source_documents")
        )
    )
    
    return rag_chain