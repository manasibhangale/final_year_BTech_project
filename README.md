# AcademIQ — AI Academic Document Intelligence System

> A local, privacy-first RAG-powered system that lets you chat with your academic documents — PDFs, DOCX, TXT, images, and URLs — using locally hosted LLMs via Ollama.

---

## What it does

AcademIQ lets you upload any academic document and ask questions about it in natural language. It retrieves the most relevant chunks from your document and grounds the LLM's answer in that context — reducing hallucinations and keeping all processing on your device.

**Supported input formats:** PDF · DOCX · TXT · Image · URL

**Features:**
- Document Q&A — ask anything about your uploaded content
- Quiz & question generation from document content
- Research paper analysis and summarisation
- Fully offline — no data sent to external servers

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Embedding Model | `all-MiniLM-L6-v2` (Sentence Transformers) |
| Vector Store | FAISS |
| LLMs | Mistral, Phi-3 Mini (via Ollama) |
| Chunking | Overlapping — 400-char chunks, 50-char overlap |
| Language | Python |

---

## Project Structure

```
AcademIQ/
├── app.py                  # Streamlit entry point
├── rag/
│   ├── chunker.py          # Overlapping text chunking
│   ├── embedder.py         # Sentence Transformer embeddings
│   ├── vector_store.py     # FAISS indexing & similarity search
│   └── llm.py              # Ollama LLM integration
├── loaders/
│   ├── pdf_loader.py
│   ├── docx_loader.py
│   ├── txt_loader.py
│   ├── image_loader.py
│   └── url_loader.py
├── features/
│   ├── qa.py               # Document Q&A
│   ├── quiz.py             # Quiz generation
│   └── analysis.py         # Research paper analysis
├── requirements.txt
└── README.md
```

---

## Prerequisites

- Python 3.9+
- [Ollama](https://ollama.com) installed and running locally
- At least one Ollama model pulled (Mistral or Phi-3 Mini recommended)

---

## Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/manasibhangale/final_year_BTech_project.git
cd final_year_BTech_project
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Pull Ollama models
```bash
ollama pull mistral
ollama pull phi3:mini
```

### 5. Start Ollama (if not already running)
```bash
ollama serve
```

---

## How to Run

```bash
streamlit run app.py
```

Then open your browser at `http://localhost:8501`

---

## How it Works

```
Document Input
      │
      ▼
Text Extraction (by format)
      │
      ▼
Overlapping Chunking (400-char / 50-char overlap)
      │
      ▼
Semantic Embedding (all-MiniLM-L6-v2)
      │
      ▼
FAISS Vector Index
      │
      ▼
User Query → Similarity Search → Top-K Chunks
      │
      ▼
Ollama LLM (Mistral / Phi-3 Mini)
      │
      ▼
Grounded Answer
```

---

## Why Local?

All inference runs on your machine via Ollama. No document content or queries are sent to any external API — making AcademIQ safe for private research, thesis work, and confidential academic material.

---

## Author

**Manasi Bhangale**
[GitHub](https://github.com/manasibhangale) · [LinkedIn](https://www.linkedin.com/in/manasi-bhangale-5878662b3/) · [Portfolio](https://manasiportfolio.vercel.app/)
