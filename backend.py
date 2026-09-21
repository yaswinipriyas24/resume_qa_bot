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
        chunk_size=400,
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

def format_docs_with_header(docs):
    """Combines retrieved chunks AND always injects the top of Page 1 (contact info) into context."""
    retrieved_text = "\n\n".join(doc.page_content for doc in docs)
    
    # Always grab Page 1 text if available so contact info, email, and summary are never missed
    header_text = ""
    if os.path.exists("temp_resume.pdf"):
        try:
            loader = PyMuPDFLoader("temp_resume.pdf")
            pages = loader.load()
            if pages:
                header_text = "--- CANDIDATE CONTACT & HEADER INFO ---\n" + pages[0].page_content + "\n----------------------------------------\n\n"
        except Exception:
            pass
            
    return header_text + retrieved_text

def get_qa_chain():
    """Initializes and returns the RAG pipeline with guaranteed contact info injection."""
    embeddings = get_embedding_model()
    
    vector_store = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is missing! Please check your .env file.")
        
    model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    
    llm = ChatGroq(
        model=model_name,
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
    
    retrieval_setup = RunnableParallel(
        context=retriever | format_docs_with_header,
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