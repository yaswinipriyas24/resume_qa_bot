import os
import streamlit as st
from operator import itemgetter
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
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

def build_vector_store(file_paths: list):
    """Loads multiple documents (PDF resumes or TXT project reports), chunks text, and stores in ChromaDB."""
    all_documents = []
    
    for path in file_paths:
        if path.endswith(".pdf"):
            loader = PyMuPDFLoader(path)
            all_documents.extend(loader.load())
        elif path.endswith(".txt") or path.endswith(".md"):
            loader = TextLoader(path)
            all_documents.extend(loader.load())
            
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(all_documents)
    
    embeddings = get_embedding_model()
    
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    return vector_store

def format_docs_with_header(docs):
    retrieved_text = "\n\n".join(doc.page_content for doc in docs)
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

def get_qa_chain(persona_mode="Recruiter Mode"):
    """Initializes and returns the RAG pipeline customized by persona."""
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
    
    if persona_mode == "Recruiter Mode":
        persona_instructions = (
            "You are an AI assistant helping a recruiter quickly understand the candidate's background.\n"
            "Summarize achievements concisely using professional, business-friendly language."
        )
    else:
        persona_instructions = (
            "You are an AI technical co-pilot representing the candidate in a technical interview.\n"
            "Provide deep technical details regarding code architectures, algorithms, hyperparameters, frameworks, and data pipelines."
        )
        
    system_prompt = (
        f"{persona_instructions}\n"
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

def analyze_job_match(job_description: str):
    """Compares the resume database against a Job Description to evaluate fit and missing skills."""
    embeddings = get_embedding_model()
    vector_store = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    # Retrieve relevant resume chunks matching the job description requirements
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    matched_docs = retriever.invoke(job_description)
    context = "\n\n".join(doc.page_content for doc in matched_docs)
    
    api_key = os.getenv("GROQ_API_KEY")
    llm = ChatGroq(
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        temperature=0.1,
        groq_api_key=api_key
    )
    
    analysis_prompt = (
        "You are an expert Applicant Tracking System (ATS) and Technical Recruiter.\n"
        "Analyze the candidate's resume context below against the provided Job Description.\n"
        "Provide:\n"
        "1. **Match Score Percentage** (e.g., 85%)\n"
        "2. **Strong Alignment / Matching Skills**\n"
        "3. **Identified Skill Gaps / Missing Requirements**\n"
        "4. **Recommendation Summary**\n\n"
        f"Job Description:\n{job_description}\n\n"
        f"Candidate Resume Context:\n{context}"
    )
    
    response = llm.invoke(analysis_prompt)
    return response.content, matched_docs