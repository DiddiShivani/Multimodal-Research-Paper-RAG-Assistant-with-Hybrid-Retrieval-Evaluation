# 🔬 Research Paper Multimodal RAG Assistant

A production-grade **Retrieval-Augmented Generation** system for academic papers, featuring multimodal ingestion, hybrid search, cross-encoder reranking, inline citations, and RAGAS evaluation — all powered by **open-source models only**.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                          frontend.py  (Streamlit)                │
│   Sidebar: API key · PDF upload · Process btn · RAGAS trigger    │
│   Main:    Chat tab  ↔  Evaluation Metrics tab                   │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                          backend.py  (RAGPipeline)               │
│                                                                  │
│  ① INGEST                                                        │
│     unstructured.io partition_pdf()                              │
│       ├── Text  →  as-is                                         │
│       ├── Tables → Groq llama-3.3-70b-versatile summarise HTML                │
│       └── Images → Groq llama-3.2-11b-vision describe figure     │
│                                                                  │
│  ② INDEX  (LangChain Indexing API)                               │
│     SQLRecordManager  →  cache hit/miss per source file          │
│     FAISS.from_documents(summaries)   ← vector store             │
│     InMemoryByteStore(originals)      ← doc store                │
│                                                                  │
│  ③ RETRIEVE  (Hybrid = FAISS + BM25 via EnsembleRetriever)       │
│     EnsembleRetriever → Reciprocal Rank Fusion (weights 0.6/0.4) │
│     CrossEncoder(BAAI/bge-reranker-base) → Top-K reranked docs   │
│                                                                  │
│  ④ GENERATE                                                      │
│     ChatGroq(llama-3.3-70b-versatile) + citation prompt                       │
│     Inline [Source N] refs + ## Sources section                  │
│                                                                  │
│  ⑤ EVALUATE  (RAGAS)                                             │
│     Faithfulness · Answer Relevancy · Context Recall          │
└──────────────────────────────────────────────────────────────────┘
```

---

## Stack

| Component | Technology |
|---|---|
| **LLM** | `ChatGroq` — `llama-3.3-70b-versatile` (text), `llama-3.2-11b-vision-preview` (images) |
| **Embeddings** | `HuggingFaceEmbeddings` — `BAAI/bge-small-en-v1.5` |
| **PDF Parsing** | `unstructured[pdf]` with table inference & image extraction |
| **Vector Store** | `FAISS` (dense retrieval) |
| **Keyword Search** | `BM25Retriever` (sparse retrieval) |
| **RRF Fusion** | `EnsembleRetriever` (LangChain) |
| **Re-ranker** | `BAAI/bge-reranker-base` (CrossEncoder) |
| **Index Cache** | `SQLRecordManager` + `index()` (LangChain Indexing API) |
| **Doc Store** | `InMemoryByteStore` (originals) |
| **Evaluation** | `RAGAS` — Faithfulness, Answer Relevancy, Context Precision |
| **UI** | `Streamlit` |

---
