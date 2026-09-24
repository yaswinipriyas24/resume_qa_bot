import os
import warnings
import shutil
import streamlit as st
from backend import build_vector_store, get_conversational_qa_chain, analyze_job_match

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*is part of.*but not documented.*")

st.set_page_config(page_title="Advanced AI Resume & ATS Bot", page_icon="💼", layout="centered")

st.title("💼 Advanced AI Resume & Portfolio System")
st.write("Chat with conversation memory, switch personas, or run an interactive ATS gap-match!")

# Sidebar for Multi-Document Upload & Settings
with st.sidebar:
    st.header("📄 Document Upload")
    uploaded_files = st.file_uploader(
        "Upload Resume (PDF) & Project Notes (TXT/MD)", 
        type=["pdf", "txt", "md"], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if st.button("Process & Index Documents"):
            with st.spinner("Processing files and building vector database..."):
                saved_paths = []
                for uploaded_file in uploaded_files:
                    file_path = uploaded_file.name
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    if file_path.endswith(".pdf"):
                        if os.path.exists("temp_resume.pdf"):
                            os.remove("temp_resume.pdf")
                        shutil.move(file_path, "temp_resume.pdf")
                        saved_paths.append("temp_resume.pdf")
                    else:
                        saved_paths.append(file_path)
                
                build_vector_store(saved_paths)
                st.success("All documents indexed successfully into ChromaDB!")
                st.session_state["db_indexed"] = True

    st.divider()
    st.header("⚙️ Recruiter Settings")
    persona_mode = st.selectbox(
        "Select Interaction Mode",
        ["Recruiter Mode", "Deep-Dive Technical Mode"]
    )

# Navigation Tabs
tab1, tab2 = st.tabs(["💬 Interactive Q&A Bot", "🎯 Interactive Skills & Gap-Matcher"])

with tab1:
    # Initialize simple string-based conversation history for safe embedding processing
    if "chat_history_text" not in st.session_state:
        st.session_state.chat_history_text = ""

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! Ask me anything about the candidate's background or projects."}
        ]

    # Display Chat History UI
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("🔍 View Unique Source Citations & Page Numbers"):
                    seen_texts = set()
                    for doc in message["sources"]:
                        if doc.page_content not in seen_texts:
                            seen_texts.add(doc.page_content)
                            page_num = doc.metadata.get("page", 0) + 1
                            st.markdown(f"**Source (Page {page_num})**")
                            st.text(doc.page_content)
                            st.divider()

    # User Query Input
    if user_query := st.chat_input("Ask a question:"):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            if not os.path.exists("./chroma_db"):
                response_text = "Please upload and process your documents in the sidebar first!"
                st.warning(response_text)
                response_sources = []
            else:
                with st.spinner(f"Analyzing in {persona_mode}..."):
                    qa_chain = get_conversational_qa_chain(persona_mode=persona_mode)
                    
                    # Pass chat history safely as a formatted text block
                    result = qa_chain.invoke({
                        "input": user_query,
                        "chat_history": st.session_state.chat_history_text
                    })
                    
                    response_text = result["answer"]
                    response_sources = result.get("source_documents", [])
                    
                    st.markdown(response_text)
                    
                    if response_sources:
                        with st.expander("🔍 View Unique Source Citations & Page Numbers"):
                            seen_texts = set()
                            for doc in response_sources:
                                if doc.page_content not in seen_texts:
                                    seen_texts.add(doc.page_content)
                                    page_num = doc.metadata.get("page", 0) + 1
                                    st.markdown(f"**Source (Page {page_num})**")
                                    st.text(doc.page_content)
                                    st.divider()
                    
        # Update text-based history log
        st.session_state.chat_history_text += f"\nHuman: {user_query}\nAI: {response_text}\n"
        
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response_text,
            "sources": response_sources
        })

with tab2:
    st.header("🎯 Interactive Skills & Job Gap-Matcher")
    st.write("Paste a target Job Description and optionally specify a focal skill to analyze alignment.")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        jd_input = st.text_area("Paste Job Description Here:", height=180)
    with col2:
        target_skill_filter = st.selectbox(
            "Interactive Skill Focus",
            ["None (General Match)", "Python", "SQL", "Tableau / Power BI", "Machine Learning / NLP", "FastAPI / React"]
        )
    
    if st.button("Run Gap-Match Analysis"):
        if not os.path.exists("./chroma_db"):
            st.warning("Please upload and index your documents in the sidebar first!")
        elif not jd_input.strip():
            st.warning("Please enter a valid job description.")
        else:
            skill_query = "" if "None" in target_skill_filter else target_skill_filter
            with st.spinner("Evaluating candidate fit and skill alignment..."):
                analysis_report, jd_sources = analyze_job_match(jd_input, target_skill=skill_query)
                st.markdown("### 📊 Skill Gap & ATS Report")
                st.markdown(analysis_report)
                
                with st.expander("🔍 View Relevant Resume Chunks"):
                    for i, doc in enumerate(jd_sources):
                        st.markdown(f"**Match Chunk {i+1}**")
                        st.text(doc.page_content)
                        st.divider()