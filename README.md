# Resume Q&A Bot

A local Streamlit application for exploring a candidate's resume with AI. Upload a resume and optional project notes, ask questions grounded in those documents, compare the resume with a job description, and edit and download an AI-drafted tailored resume as a PDF.

## Features

- **Document upload and indexing:** Upload one or more PDF resumes and TXT or Markdown notes. The app extracts text, splits it into overlapping chunks, creates embeddings, and stores them in a local Chroma vector database.
- **Resume Q&A:** Ask questions about the indexed content and view the retrieved source passages with page numbers for PDF pages.
- **Two answer styles:** Choose **Recruiter Mode** for concise candidate summaries or **Deep-Dive Technical Mode** for detailed technical explanations.
- **Job description matching:** Paste a job description to receive an AI-generated match percentage, matching skills, gaps, and (optionally) a focused skill evaluation.
- **Resume tailoring:** Generate a role-focused resume draft based on retrieved resume content, edit the draft in the app, and download it as a PDF.
- **Skill focus:** The matching screen offers general matching or focus areas for Python, SQL, Tableau / Power BI, Machine Learning / NLP, and SAP BTP / Teaching.

## How it works

1. The Streamlit interface in `app.py` accepts documents and user input.
2. `backend.py` loads PDF files with PyMuPDF and text files with LangChain's text loader.
3. Documents are split into 400-character chunks with 100 characters of overlap. The `all-MiniLM-L6-v2` Hugging Face embedding model converts chunks and search queries into vectors.
4. Chroma stores and retrieves relevant chunks from `./chroma_db` (four results for Q&A and five for job matching and tailoring).
5. The Groq chat model generates answers and reports from the retrieved context. The default model is `openai/gpt-oss-120b`; it can be changed with `GROQ_MODEL`.
6. The PDF export converts the edited text to a simple PDF using FPDF.

Q&A prompts ask the model to answer from retrieved context and chat history and to say when the context does not contain an answer. Q&A history and screen state are held in the current Streamlit session. The vector database is persisted locally, so it can remain between application runs.

## Requirements

- Python 3.10 or newer is recommended.
- A Groq API key, configured as `GROQ_API_KEY`.
- Internet access on first run to download the embedding model and to call Groq.

The dependencies are listed in `requirements.txt`. The PDF export code also imports `fpdf`, so install **fpdf2** as shown below; it is not currently listed in `requirements.txt`.

## Setup

Run these commands from the project directory.

### 1. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

macOS or Linux:

```bash
python -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install fpdf2
```

### 3. Configure the Groq API key

Create a file named `.env` in the project directory:

```dotenv
GROQ_API_KEY=your_groq_api_key
# Optional; defaults to openai/gpt-oss-120b
GROQ_MODEL=openai/gpt-oss-120b
```

Keep `.env` private. It is excluded from Git by `.gitignore`.

### 4. Start the app

```bash
streamlit run app.py
```

Streamlit prints a local URL in the terminal (usually `http://localhost:8501`). Open that URL in a browser.

## Using the application

### Upload and index documents

1. In the sidebar, upload a PDF resume and any relevant TXT or Markdown notes.
2. Select **Process & Index Documents** and wait for indexing to finish.
3. Use the Q&A tab or the job matching tab.

Process the documents before asking questions or matching a job. Indexing creates or updates the local `chroma_db` directory. The uploaded PDF is copied to the fixed filename `temp_resume.pdf`; a subsequent PDF upload replaces that file. Text and Markdown uploads are saved using their uploaded filenames in the working directory.

### Ask resume questions

In **Interactive Q&A Bot**, enter a question in the chat box. Choose the interaction mode in the sidebar. Expand the source citation section under an answer to inspect retrieved text; page numbers are shown for PDF sources.

### Match and tailor for a job

1. Open **Skills Gap-Matcher & Auto-Tailor**.
2. Paste a job description and select a skill focus, if desired.
3. Choose **Run ATS Gap-Match & Analysis** to view the generated report and relevant resume chunks.
4. Choose **Generate Tailored Resume**, review and edit the draft, then download the PDF.

The match percentage, gap analysis, and tailored text are model-generated suggestions. Review them for accuracy before relying on them or sharing a tailored resume.

## Project files

| File or directory | Purpose |
| --- | --- |
| `app.py` | Streamlit interface, uploads, chat, job match controls, editing, and download flow. |
| `backend.py` | Document ingestion, embeddings, Chroma retrieval, Groq prompts, and PDF generation. |
| `requirements.txt` | Python package dependencies (see the `fpdf2` note above). |
| `.env` | Local Groq credentials and optional model setting; do not commit this file. |
| `chroma_db/` | Generated local vector database; excluded from Git. |
| `temp_resume.pdf` | Working copy of the most recently uploaded PDF resume; excluded from Git. |
| `venv/`, `__pycache__/` | Local Python environment and generated bytecode; excluded from Git. |

## Data and privacy

Uploaded files and the vector database are handled in the local project directory. Retrieved resume text and prompts are sent to the configured Groq API to generate answers, analyses, and tailored drafts. The Hugging Face embedding model may be downloaded on first use. Do not upload documents unless you are permitted to process them this way, and avoid committing resumes, `.env`, `chroma_db/`, or other private data.

## Troubleshooting

- **`GROQ_API_KEY is missing`:** Add a valid `GROQ_API_KEY` entry to `.env` in the project directory, then restart Streamlit.
- **No resume has been indexed:** Upload documents and select **Process & Index Documents** before using Q&A or job matching.
- **PDF export import error:** Install the dependency with `python -m pip install fpdf2`.
- **Model download or API errors:** Check network access, the Groq key, and (if set) the `GROQ_MODEL` value.
- **Stale or unwanted indexed documents:** Stop the app and remove the generated `chroma_db/` directory, then upload and index the intended documents again. This deletes the local index.
