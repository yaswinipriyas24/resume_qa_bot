import os
import warnings
import streamlit as st
from backend import build_vector_store, get_qa_chain

# Suppress harmless Hugging Face library warnings in terminal
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*is part of.*but not documented.*")

st.set_page_config(page_title="AI Resume QA Bot", page_icon="💼", layout="centered")

st.title("💼 AI Portfolio & Resume QA Bot")
st.write("Upload your resume (PDF) and ask questions with live source citations!")

# Sidebar for PDF Upload
with st.sidebar:
    st.header("📄 Upload Document")
    uploaded_file = st.file_uploader("Upload your PDF Resume", type=["pdf"])
    
    if uploaded_file is not None:
        if st.button("Process & Index Resume"):
            with st.spinner("Chunking text and creating vector embeddings..."):
                temp_pdf_path = "temp_resume.pdf"
                with open(temp_pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                build_vector_store(temp_pdf_path)
                st.success("Resume successfully indexed into ChromaDB!")
                st.session_state["db_indexed"] = True

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Ask me anything about the candidate's skills, projects, or background."}
    ]

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # If stored message has sources, render them in history with deduplication
        if "sources" in message and message["sources"]:
            with st.expander("🔍 View Source Citations & Page Numbers"):
                seen_texts = set()
                for doc in message["sources"]:
                    if doc.page_content not in seen_texts:
                        seen_texts.add(doc.page_content)
                        page_num = doc.metadata.get("page", 0) + 1  # PyMuPDF is 0-indexed
                        st.markdown(f"**Source (Page {page_num})**")
                        st.text(doc.page_content)
                        st.divider()

# User Query Input
if user_query := st.chat_input("Ask a question (e.g., 'What projects were built with Python?'):"):
    # Add user message to UI and history
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Generate Response
    with st.chat_message("assistant"):
        if not os.path.exists("./chroma_db"):
            response_text = "Please upload and process a resume PDF in the sidebar first!"
            st.warning(response_text)
            response_sources = []
        else:
            with st.spinner("Searching vector database and verifying sources..."):
                qa_chain = get_qa_chain()
                # Invoking the chain returns a dict: {"answer": ..., "source_documents": ...}
                result = qa_chain.invoke(user_query)
                response_text = result["answer"]
                response_sources = result["source_documents"]
                
                st.markdown(response_text)
                
                # Display Unique Source Citations directly under the new message
                if response_sources:
                    with st.expander("🔍 View Source Citations & Page Numbers"):
                        seen_texts = set()
                        for doc in response_sources:
                            if doc.page_content not in seen_texts:
                                seen_texts.add(doc.page_content)
                                page_num = doc.metadata.get("page", 0) + 1
                                st.markdown(f"**Source (Page {page_num})**")
                                st.text(doc.page_content)
                                st.divider()
                
    # Save response and sources to session history
    st.session_state.messages.append({
        "role": "assistant", 
        "content": response_text,
        "sources": response_sources
    })