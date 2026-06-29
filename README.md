# RAG Chatbot — Document Q&A with Hallucination Detection

A production-style Retrieval-Augmented Generation (RAG) pipeline that answers questions over custom documents using LLMs, with automatic faithfulness evaluation using BERTScore.

## What it does
- Ingests PDF documents and builds a semantic search index
- Answers natural language questions using only the document content
- Refuses out-of-scope questions to prevent hallucination
- Scores every answer for faithfulness using BERTScore (F1)

## Architecture

PDF -> Chunking -> Embeddings -> FAISS Index

Query -> Embed -> Retrieve top-k chunks -> LLM -> Answer + BERTScore

## Tech stack

| Component | Tool |
|---|---|
| LLM | Llama 3.3 70B via Groq |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector store | FAISS |
| Framework | LangChain LCEL |
| Evaluation | BERTScore |
| UI | Streamlit |

## Setup

    git clone https://github.com/adityabandal21/rag-chatbot.git
    cd rag-chatbot
    python3.11 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

Add your Groq API key to .env:

    GROQ_API_KEY=your_key_here

## Run

Step 1 - Ingest your documents:

    python ingest.py

Step 2 - Launch the app:

    python -m streamlit run app.py

## Key design decisions
- Chunk size 500, overlap 50 — balances context preservation with retrieval precision
- top-k=3 retrieval — enough context without exceeding prompt limits
- BERTScore F1 above 0.85 = high faithfulness, 0.75-0.85 = moderate, below 0.75 = low
- Prompt guardrail — LLM explicitly instructed to refuse out-of-scope questions

## Skills demonstrated
RAG pipelines, LangChain LCEL, Vector databases, LLM evaluation, Hallucination detection, Responsible AI, HuggingFace, Streamlit deployment
