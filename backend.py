"""
backend.py — Research Paper Multimodal RAG Pipeline
=====================================================
Handles: PDF ingestion, multimodal summarization, hybrid retrieval,
         re-ranking, citation-aware generation, and RAGAS evaluation.
"""

from __future__ import annotations

import os
import uuid
import base64
import pickle
import tempfile
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv

# ── LangChain core ────────────────────────────────────────────────────────────
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

# ── LangChain integrations ────────────────────────────────────────────────────
# NOTE: LangChain v1.x relocated several modules to `langchain_classic`.
#       All imports below are verified against langchain==1.x / langchain_classic==1.x
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever

# EnsembleRetriever → langchain_classic in v1.x
from langchain_classic.retrievers import EnsembleRetriever

# MultiVectorRetriever → langchain_classic in v1.x
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever

# InMemoryByteStore → langchain_core.stores in v1.x
from langchain_core.stores import InMemoryByteStore

# ── LangChain Indexing API (cache hit/miss) ────────────────────────────────────
# SQLRecordManager + index() → langchain_classic.indexes in v1.x
from langchain_classic.indexes import SQLRecordManager, index

# ── Unstructured PDF partitioning ─────────────────────────────────────────────
from unstructured.partition.pdf import partition_pdf

# ── Cross-encoder re-ranker ────────────────────────────────────────────────────
from sentence_transformers import CrossEncoder

# ── RAGAS evaluation ───────────────────────────────────────────────────────────
from datasets import Dataset

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_EMBED_MODEL    = "BAAI/bge-small-en-v1.5"
DEFAULT_TEXT_LLM       = "llama-3.3-70b-versatile" 
DEFAULT_VISION_LLM     = "llama-3.2-11b-vision-preview"
DEFAULT_RERANKER       = "BAAI/bge-reranker-base"
RECORD_MANAGER_DB       = "sqlite:///record_manager_cache.db"

# ─────────────────────────────────────────────────────────────────────────────
# Element-type helpers
# ─────────────────────────────────────────────────────────────────────────────
def _is_table(el) -> bool:
    return type(el).__name__ == "Table"

def _is_image(el) -> bool:
    return type(el).__name__ == "Image"

def _is_text(el) -> bool:
    return type(el).__name__ in {
        "NarrativeText", "Title", "CompositeElement",
        "ListItem", "Header", "Text", "FigureCaption",
    }

# ─────────────────────────────────────────────────────────────────────────────
# RAGPipeline
# ─────────────────────────────────────────────────────────────────────────────
class RAGPipeline:
    """
    End-to-end Multimodal RAG pipeline.

    Pipeline stages
    ---------------
    1. Ingest   → partition_pdf → categorise text / table / image
    2. Summarise → Groq LLM (text) | Groq Vision LLM (image) | LLM (table)
    3. Index    → FAISS + RecordManager (cache hit/miss)
    4. Retrieve → EnsembleRetriever (FAISS + BM25) → RRF fusion
    5. Rerank   → CrossEncoder (BAAI/bge-reranker-base)
    6. Generate → ChatGroq with inline [Source N] citations
    7. Evaluate → RAGAS (Faithfulness, Answer Relevancy, Context Recall)
    """

    def __init__(
        self,
        groq_api_key: str,
        embed_model: str = DEFAULT_EMBED_MODEL,
        text_llm_model: str = DEFAULT_TEXT_LLM,
        vision_llm_model: str = DEFAULT_VISION_LLM,
        reranker_model: str = DEFAULT_RERANKER,
        top_k: int = 5,
        ensemble_weights: Tuple[float, float] = (0.6, 0.4),
    ) -> None:
        self.top_k = top_k
        self.ensemble_weights = ensemble_weights

        # ── LLMs ────────────────────────────────────────────────────────────
        self.llm = ChatGroq(
            api_key=groq_api_key,
            model=text_llm_model,
            temperature=0,
        )
        self.vision_llm = ChatGroq(
            api_key=groq_api_key,
            model=vision_llm_model,
            temperature=0,
        )

        # ── Embeddings ───────────────────────────────────────────────────────
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embed_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        # ── Cross-encoder reranker ────────────────────────────────────────────
        logger.info("Loading cross-encoder: %s", reranker_model)
        self.cross_encoder = CrossEncoder(reranker_model)

        # ── Storage ──────────────────────────────────────────────────────────
        self.docstore        = InMemoryByteStore()
        self.vectorstore: Optional[FAISS] = None
        self._all_summary_docs: List[Document] = []  # kept for BM25

        # ── State flags ───────────────────────────────────────────────────────
        self._ensemble_retriever: Optional[EnsembleRetriever] = None
        self.is_ready = False

    # =========================================================================
    # INGESTION
    # =========================================================================

    def _partition_pdf(
        self,
        pdf_path: str,
        output_dir: str,
        cb: Optional[Callable] = None,
    ) -> Tuple[list, list, list]:
        """
        Run unstructured partition_pdf and split results into
        (text_elements, table_elements, image_elements).
        """
        _cb(cb, 0.08, "📄 Partitioning PDF with unstructured.io …")

        elements = partition_pdf(
            filename=pdf_path,
            extract_images_in_pdf=True,
            infer_table_structure=True,
            chunking_strategy="by_title",
            max_characters=4000,
            new_after_n_chars=3800,
            combine_text_under_n_chars=2000,
            image_output_dir_path=output_dir,
        )

        texts, tables, images = [], [], []
        for el in elements:
            if _is_table(el):
                tables.append(el)
            elif _is_image(el):
                images.append(el)
            elif _is_text(el) and el.text.strip():
                texts.append(el)
            elif hasattr(el, "text") and el.text.strip():
                texts.append(el)

        _cb(
            cb, 0.22,
            f"✅ Found {len(texts)} text  |  {len(tables)} tables  |  {len(images)} images",
        )
        return texts, tables, images

    # ── Summarizers ──────────────────────────────────────────────────────────

    def _summarize_table(self, table_html: str) -> str:
        """Groq LLM → descriptive summary of an HTML table."""
        prompt = ChatPromptTemplate.from_template(
            "You are an expert research analyst.\n"
            "Provide a concise, informative summary of the following HTML table "
            "extracted from a research paper. Focus on key findings, compared methods, "
            "numerical results, and scientific significance.\n\n"
            "TABLE HTML:\n{table}\n\nSUMMARY:"
        )
        chain = prompt | self.llm | StrOutputParser()
        try:
            return chain.invoke({"table": table_html})
        except Exception as exc:
            logger.warning("Table summarisation failed: %s", exc)
            return table_html[:500]

    def _summarize_image(self, image_b64: str) -> str:
        """Groq Vision LLM → descriptive caption for a figure/image."""
        try:
            msg = HumanMessage(
                content=[
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                    {
                        "type": "text",
                        "text": (
                            "You are an expert research analyst. "
                            "Describe this figure from a research paper in detail: "
                            "what is depicted, axis labels, trends, key data points, "
                            "and its scientific significance."
                        ),
                    },
                ]
            )
            return self.vision_llm.invoke([msg]).content
        except Exception as exc:
            logger.warning("Image summarisation failed: %s", exc)
            return "[Figure: description unavailable]"

    # ── Main process entry point ──────────────────────────────────────────────

    def process_pdf(
        self,
        pdf_path: str,
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Full ingestion → indexing pipeline for one PDF.
        Returns stats dict.
        """
        source_name = Path(pdf_path).name
        cb = progress_callback

        with tempfile.TemporaryDirectory() as tmpdir:
            # ── 1. Partition ─────────────────────────────────────────────────
            text_els, table_els, image_els = self._partition_pdf(
                pdf_path, tmpdir, cb
            )

            # ── 2. Build (doc_id, original_doc) pairs + summary docs ─────────
            all_ids:      List[str]              = []
            all_originals: List[Document]        = []
            all_summaries: List[Document]        = []

            # — Text —
            _cb(cb, 0.30, "📝 Processing text chunks …")
            for i, el in enumerate(text_els):
                doc_id = str(uuid.uuid4())
                meta   = _meta(source_name, "text", doc_id, i)
                orig   = Document(page_content=el.text, metadata=meta)
                summ   = Document(page_content=el.text, metadata=meta)   # text IS the summary
                all_ids.append(doc_id)
                all_originals.append(orig)
                all_summaries.append(summ)

            # — Tables —
            _cb(cb, 0.40, f"📊 Summarising {len(table_els)} tables with LLM …")
            for i, el in enumerate(table_els):
                doc_id    = str(uuid.uuid4())
                table_html = getattr(el.metadata, "text_as_html", el.text) or el.text
                summary   = self._summarize_table(table_html)
                meta      = _meta(source_name, "table", doc_id, i)
                orig      = Document(page_content=table_html, metadata=meta)
                summ      = Document(page_content=summary,    metadata=meta)
                all_ids.append(doc_id)
                all_originals.append(orig)
                all_summaries.append(summ)

            # — Images —
            _cb(cb, 0.55, f"🖼️  Summarising {len(image_els)} images with Vision LLM …")
            for i, el in enumerate(image_els):
                doc_id  = str(uuid.uuid4())
                img_b64 = getattr(el.metadata, "image_base64", None)
                summary = self._summarize_image(img_b64) if img_b64 else f"[Figure {i+1}]"
                meta    = _meta(source_name, "image", doc_id, i)
                orig    = Document(
                    page_content=img_b64 or f"Figure {i+1}",
                    metadata={**meta, "summary": summary},
                )
                summ    = Document(page_content=summary, metadata=meta)
                all_ids.append(doc_id)
                all_originals.append(orig)
                all_summaries.append(summ)

            # ── 3. Persist originals in docstore ─────────────────────────────
            self.docstore.mset(
                [(did, pickle.dumps(doc))
                 for did, doc in zip(all_ids, all_originals)]
            )

            # ── 4. Index summaries → FAISS with RecordManager cache ──────────
            _cb(cb, 0.68, "🗂️  Indexing embeddings (cache-aware) …")
            cache_stats = self._index_summaries(all_summaries, source_name, cb)

            # ── 5. Rebuild ensemble retriever ────────────────────────────────
            _cb(cb, 0.82, "🔗 Building hybrid retriever (FAISS + BM25) …")
            self._build_ensemble_retriever()

            _cb(cb, 1.00, "✅ Pipeline ready!")
            self.is_ready = True

            return {
                "source":      source_name,
                "text_count":  len(text_els),
                "table_count": len(table_els),
                "image_count": len(image_els),
                "cache_added":   cache_stats.get("num_added", 0),
                "cache_skipped": cache_stats.get("num_skipped", 0),
            }

    # =========================================================================
    # INDEXING  (LangChain Indexing API — cache hit / miss)
    # =========================================================================

    def _index_summaries(
        self,
        summary_docs: List[Document],
        source_name: str,
        cb: Optional[Callable] = None,
    ) -> Dict[str, int]:
        """
        Insert summary docs into FAISS using RecordManager for deduplication.
        Returns index() stats dict.
        """
        record_manager = SQLRecordManager(
            namespace=f"faiss/{source_name}",
            db_url=RECORD_MANAGER_DB,
        )
        record_manager.create_schema()

        if self.vectorstore is None:
            # First-time init
            self.vectorstore = FAISS.from_documents(summary_docs[:1], self.embeddings)
            if len(summary_docs) > 1:
                self.vectorstore.add_documents(summary_docs[1:])
            self._all_summary_docs.extend(summary_docs)
            return {"num_added": len(summary_docs), "num_skipped": 0}

        # Incremental indexing with cache
        stats = index(
            summary_docs,
            record_manager,
            self.vectorstore,
            cleanup="incremental",
            source_id_key="source",
        )
        # Only extend BM25 corpus with newly added docs
        added = stats.get("num_added", 0)
        if added:
            self._all_summary_docs.extend(summary_docs[:added])

        if cb:
            cb(
                0.75,
                f"📦 Cache: {stats.get('num_added',0)} added, "
                f"{stats.get('num_skipped',0)} skipped (already indexed).",
            )
        return stats

    # =========================================================================
    # RETRIEVAL  (Hybrid FAISS + BM25 → Reciprocal Rank Fusion)
    # =========================================================================

    def _build_ensemble_retriever(self) -> None:
        """
        (Re)builds the EnsembleRetriever combining FAISS and BM25.
        EnsembleRetriever implements Reciprocal Rank Fusion internally.
        """
        if self.vectorstore is None or not self._all_summary_docs:
            return

        faiss_ret = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 10},
        )

        bm25_ret = BM25Retriever.from_documents(self._all_summary_docs)
        bm25_ret.k = 10

        self._ensemble_retriever = EnsembleRetriever(
            retrievers=[faiss_ret, bm25_ret],
            weights=list(self.ensemble_weights),
        )

    def _hybrid_retrieve(self, query: str) -> List[Document]:
        """Run hybrid retrieval (FAISS + BM25 with RRF)."""
        if self._ensemble_retriever is None:
            return []
        return self._ensemble_retriever.invoke(query)

    # ── Cross-encoder reranking ───────────────────────────────────────────────

    def rerank(
        self,
        query: str,
        docs: List[Document],
        top_k: Optional[int] = None,
    ) -> List[Document]:
        """Re-rank docs with BAAI/bge-reranker-base cross-encoder."""
        if not docs:
            return []
        k     = top_k or self.top_k
        pairs = [(query, doc.page_content) for doc in docs]
        scores = self.cross_encoder.predict(pairs)

        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        result = []
        for rank, (score, doc) in enumerate(ranked[:k], start=1):
            doc.metadata["rerank_score"] = float(score)
            doc.metadata["rank"]         = rank
            result.append(doc)
        return result

    def retrieve_and_rerank(self, query: str) -> List[Document]:
        """Full retrieval pipeline: hybrid → rerank → top-k."""
        candidates = self._hybrid_retrieve(query)
        return self.rerank(query, candidates)

    # =========================================================================
    # GENERATION  (with inline citations)
    # =========================================================================

    @staticmethod
    def _format_context(docs: List[Document]) -> str:
        """Render retrieved docs with [Source N] headers for the prompt."""
        parts = []
        for i, doc in enumerate(docs, 1):
            m = doc.metadata
            header = (
                f"[Source {i}] "
                f"File: {m.get('source','?')}  |  "
                f"Type: {m.get('type','text')}  |  "
                f"Element #{m.get('element_index','?')}  |  "
                f"Rerank score: {m.get('rerank_score', 0):.3f}"
            )
            parts.append(f"{header}\n{doc.page_content}")
        return "\n\n" + ("─" * 60 + "\n\n").join(parts)

    _ANSWER_PROMPT = ChatPromptTemplate.from_template(
        """You are an expert research assistant that analyses academic papers.

Answer the question ONLY from the provided context. Be thorough and precise.

RULES
─────
1. Use inline citations: e.g. "The model achieves 94 % accuracy [Source 1]."
2. For tables / figures note the type: "As shown in Table [Source 2] …"
3. End your answer with a "## Sources" section that lists every cited source
   with its file, type, and element number.
4. If the context is insufficient, state so clearly.

CONTEXT
───────
{context}

QUESTION
────────
{question}

ANSWER
──────"""
    )

    def answer(self, question: str) -> Dict[str, Any]:
        """
        Retrieve → rerank → generate answer with citations.
        Returns dict with 'answer', 'sources', 'context'.
        """
        if not self.is_ready:
            return {
                "answer":  "⚠️ Pipeline not initialised. Please process a PDF first.",
                "sources": [],
                "context": "",
            }

        docs    = self.retrieve_and_rerank(question)
        context = self._format_context(docs)

        chain   = self._ANSWER_PROMPT | self.llm | StrOutputParser()
        text    = chain.invoke({"context": context, "question": question})

        return {"answer": text, "sources": docs, "context": context}

    # =========================================================================
    # EVALUATION  (RAGAS)
    # =========================================================================

    def evaluate(self, questions: List[str]) -> Dict[str, Any]:
        """
        Run RAGAS evaluation metrics:
          • Faithfulness
          • Answer Relevancy
          • Context Precision

        Returns a dict of metric lists (one value per question).
        """
        if not self.is_ready:
            return {"error": "Pipeline not ready — process a PDF first."}

        # ── Collect answers & contexts ────────────────────────────────────────
        answers:  List[str]        = []
        contexts: List[List[str]]  = []

        for q in questions:
            res = self.answer(q)
            answers.append(res["answer"])
            contexts.append([d.page_content for d in res["sources"]])

        # ── Build HuggingFace Dataset ─────────────────────────────────────────
        eval_ds = Dataset.from_dict({
            "question": questions,
            "answer":   answers,
            "contexts": contexts,
            "reference": [""] * len(questions),
        })

        # ── Wire RAGAS to our LLM & embeddings ──────────────────────────────
        try:
            from ragas import evaluate as ragas_eval
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_recall,
            )
            from ragas.llms      import LangchainLLMWrapper
            from ragas.embeddings import LangchainEmbeddingsWrapper

            ragas_llm   = LangchainLLMWrapper(self.llm)
            ragas_emb   = LangchainEmbeddingsWrapper(self.embeddings)
            metrics     = [faithfulness, answer_relevancy, context_recall]

            for m in metrics:
                m.llm = ragas_llm
                if hasattr(m, "embeddings"):
                    m.embeddings = ragas_emb

            result = ragas_eval(dataset=eval_ds, metrics=metrics)
            return result.to_pandas().to_dict(orient="list")

        except Exception as exc:
            logger.error("RAGAS evaluation failed: %s", exc)
            return {"error": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cb(fn: Optional[Callable], val: float, msg: str) -> None:
    """Safe progress-callback invoker."""
    if fn:
        try:
            fn(val, msg)
        except Exception:
            pass


def _meta(source: str, kind: str, doc_id: str, idx: int) -> Dict[str, Any]:
    return {
        "source":        source,
        "type":          kind,
        "doc_id":        doc_id,
        "element_index": idx,
    }