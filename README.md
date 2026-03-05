# Enterprise Policy RAG System

## Overview

This project implements a Retrieval-Augmented Generation (RAG) system for answering enterprise policy questions strictly from official documents. The system ensures no hallucination by returning only retrieved content with verifiable citations and a confidence score.

It is designed for HR, IT, Security, and Compliance policy environments where accuracy and traceability are critical.

---



### 1. Document Ingestion

* 50–100 policy PDFs
* Text extracted using PyMuPDF
* Metadata captured: source, page, policy type, version

### 2. Chunking

* Paragraph-based chunking
* Chunk size: 800
* No overlap
* Structured separators to preserve policy integrity

### 3. Embeddings

* Model: all-MiniLM-L6-v2
* Vector embeddings stored in persistent ChromaDB
* Embeddings rebuilt when chunking or model changes

### 4. Retrieval

* Vector similarity search (top-k)
* Distance-based scoring
* Configurable relevance threshold
* Automatic refusal if similarity is weak

### 5. Answer Generation

* No LLM-based generation
* Returns retrieved paragraph directly
* Mandatory citation included
* Confidence score provided

---

## Output Format

FINAL ANSWER
Retrieved policy text

SOURCE
Document name + page

CONFIDENCE
Score between 0 and 1

---

## Assumptions

* Policy documents are authoritative and well-structured.
* Users ask questions related to existing policy content.
* Similarity threshold is tuned for precision over recall.
* System prioritizes correctness over conversational flexibility.

---

## Hallucination Control

* No answer without retrieval
* No free-text generation
* Refusal if below relevance threshold
* Citation required for every answer

---

## Evaluation Focus

* Retrieval accuracy
* Citation correctness
* Proper refusal handling
* Confidence score reliability

---


project/
│
├── main.py                  # Entry point — build or chat mode
├── data/                    # Drop your PDF files here
├── vector_db/               # Auto-created — ChromaDB persistent store
│
└── src/
    ├── config.py            # All tunable settings (chunk size, models, threshold)
    ├── loader.py            # Scans data/ and loads PDFs page by page
    ├── chunker.py           # Splits pages into overlapping text chunks
    ├── embedder.py          # Batch-embeds chunks and inserts into ChromaDB
    ├── retriever.py         # Embeds query, fetches top-K similar chunks
    ├── llm.py               # Singleton ChatOllama instance (llama3)
    └── rag.py               # Reranking, filtering, prompting, answer extraction