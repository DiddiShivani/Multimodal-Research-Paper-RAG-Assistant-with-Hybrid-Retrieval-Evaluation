"""
frontend.py — Research Paper Multimodal RAG Assistant
======================================================
Streamlit UI with:
  • Sidebar  : API-key input, PDF upload, Process button (with progress bar),
               RAGAS evaluation trigger
  • Main     : Chat tab + Evaluation Metrics tab
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from backend import RAGPipeline

# ─────────────────────────────────────────────────────────────────────────────
# Page config  (MUST be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Research RAG Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Global CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* ── Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
code, pre { font-family: 'JetBrains Mono', monospace; font-size: 0.82em; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #21262d;
}
section[data-testid="stSidebar"] * { color: #e6edf3 !important; }
section[data-testid="stSidebar"] .stTextInput input {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 6px;
    color: #e6edf3 !important;
}

/* ── Main background ── */
.main { background: #0d1117; color: #e6edf3; }

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    margin-bottom: 10px;
    padding: 14px;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 16px;
}
[data-testid="stMetricValue"] { color: #58a6ff !important; font-size: 2rem !important; }
[data-testid="stMetricLabel"] { color: #8b949e !important; }

/* ── Buttons ── */
.stButton > button {
    background: #238636;
    color: #ffffff !important;
    border: none;
    border-radius: 6px;
    font-weight: 600;
    transition: background 0.2s;
}
.stButton > button:hover { background: #2ea043; }

/* ── Tab headers ── */
button[data-baseweb="tab"] { font-weight: 600; font-size: 1rem; }
button[data-baseweb="tab"][aria-selected="true"] { color: #58a6ff !important; }

/* ── Expander ── */
details { background: #161b22; border-radius: 8px; border: 1px solid #21262d; }

/* ── Source badges ── */
.source-badge {
    display: inline-block;
    background: #1f6feb22;
    color: #58a6ff;
    border: 1px solid #1f6feb;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.78em;
    font-family: 'JetBrains Mono', monospace;
    margin-right: 6px;
}

/* ── Progress bar ── */
.stProgress > div > div { background: #238636; }

/* ── Info/success/error boxes ── */
.stAlert { border-radius: 8px; }

/* ── DataFrame ── */
[data-testid="stDataFrame"] { background: #161b22; border-radius: 8px; }
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Session-state defaults
# ─────────────────────────────────────────────────────────────────────────────
_DEFAULTS = {
    "pipeline":        None,
    "chat_history":    [],
    "processed_files": [],
    "process_stats":   [],
    "eval_results":    None,
}
for key, val in _DEFAULTS.items():
    st.session_state.setdefault(key, val)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🔬 Research RAG")
    st.caption("Multimodal · Hybrid Search · Citations")
    st.divider()

    # ── API Key ───────────────────────────────────────────────────────────────
    st.subheader("🔑 API Configuration")
    groq_api_key = st.text_input(
        "Groq API Key",
        type="password",
        value=os.getenv("GROQ_API_KEY", ""),
        placeholder="gsk_…",
        help="Free key at console.groq.com",
    )

    st.divider()

    # ── PDF Upload ────────────────────────────────────────────────────────────
    st.subheader("📄 Upload Research Paper")
    uploaded_file = st.file_uploader(
        "ArXiv PDF",
        type=["pdf"],
        label_visibility="collapsed",
        help="Upload any research paper in PDF format.",
    )

    # ── Advanced options (collapsed) ─────────────────────────────────────────
    with st.expander("⚙️ Advanced Options"):
        embed_model    = st.selectbox(
            "Embedding Model",
            ["BAAI/bge-small-en-v1.5", "BAAI/bge-base-en-v1.5", "BAAI/bge-large-en-v1.5"],
            index=0,
        )
        text_llm       = st.selectbox(
            "Text LLM",
            ["llama-3.3-70b-versatile", "llama-3.2-70b-preview", "llama-3.2-13b-preview"],
            index=0,
        )
        vision_llm     = st.selectbox(
            "Vision LLM",
            ["llama-3.2-11b-vision-preview", "llama-3.2-90b-vision-preview"],
            index=0,
        )
        reranker_model = st.selectbox(
            "Re-ranker",
            ["BAAI/bge-reranker-base", "BAAI/bge-reranker-large"],
            index=0,
        )
        top_k = st.slider("Top-K after reranking", 3, 10, 5)
        bm25_w = st.slider("BM25 weight (FAISS = 1-w)", 0.1, 0.9, 0.4, 0.05)

    # ── Process Button ────────────────────────────────────────────────────────
    process_disabled = not (uploaded_file and groq_api_key)
    if st.button(
        "🚀 Process Paper",
        use_container_width=True,
        type="primary",
        disabled=process_disabled,
    ):
        _progress_bar  = st.progress(0.0)
        _status_text   = st.empty()

        def _update(val: float, msg: str) -> None:
            _progress_bar.progress(min(val, 1.0))
            _status_text.markdown(f"*{msg}*")

        # Save upload to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name

        try:
            # Lazy init or reuse pipeline
            if st.session_state.pipeline is None:
                _update(0.02, "Initialising models …")
                st.session_state.pipeline = RAGPipeline(
                    groq_api_key=groq_api_key,
                    embed_model=embed_model,
                    text_llm_model=text_llm,
                    vision_llm_model=vision_llm,
                    reranker_model=reranker_model,
                    top_k=top_k,
                    ensemble_weights=(1 - bm25_w, bm25_w),
                )

            stats = st.session_state.pipeline.process_pdf(
                tmp_path, progress_callback=_update
            )
            st.session_state.processed_files.append(uploaded_file.name)
            st.session_state.process_stats.append(stats)

            _status_text.empty()
            st.success(
                f"**{stats['source']}** processed!\n\n"
                f"- 📝 Text chunks: **{stats['text_count']}**\n"
                f"- 📊 Tables: **{stats['table_count']}**\n"
                f"- 🖼️  Images: **{stats['image_count']}**\n"
                f"- ♻️  Cached (skipped): **{stats['cache_skipped']}**"
            )

        except Exception as exc:
            _status_text.empty()
            st.error(f"❌ Processing failed:\n\n`{exc}`")
        finally:
            os.unlink(tmp_path)

    # ── Processed files list ──────────────────────────────────────────────────
    if st.session_state.processed_files:
        st.divider()
        st.subheader("📚 Loaded Papers")
        for fname in st.session_state.processed_files:
            st.markdown(f"✅ `{fname}`")

    st.divider()

    # ── RAGAS Evaluation panel ────────────────────────────────────────────────
    st.subheader("📊 RAGAS Evaluation")
    eval_questions_raw = st.text_area(
        "Evaluation questions (one per line)",
        placeholder=(
            "What is the main contribution of this paper?\n"
            "What dataset was used for experiments?\n"
            "What are the key numerical results?\n"
            "How does this compare to prior work?"
        ),
        height=130,
        label_visibility="collapsed",
    )

    if st.button("▶  Run Evaluation", use_container_width=True):
        pipeline_ready = (
            st.session_state.pipeline is not None
            and st.session_state.pipeline.is_ready
        )
        if not pipeline_ready:
            st.warning("⚠️ Process a PDF first.")
        else:
            eval_qs = [q.strip() for q in eval_questions_raw.splitlines() if q.strip()]
            if not eval_qs:
                st.warning("Enter at least one question.")
            else:
                with st.spinner("⏳ Running RAGAS evaluation …"):
                    st.session_state.eval_results = (
                        st.session_state.pipeline.evaluate(eval_qs)
                    )
                if "error" not in st.session_state.eval_results:
                    st.success("✅ Evaluation complete — see Metrics tab.")
                else:
                    st.error(f"Evaluation error: {st.session_state.eval_results['error']}")

    # ── Footer ────────────────────────────────────────────────────────────────
    st.divider()
    st.caption(
        "🤖 Groq · 🤗 HuggingFace · 🦜 LangChain\n"
        "📐 FAISS + BM25 · ✂️ BGE-Reranker · 📏 RAGAS"
    )


def _render_sources(sources) -> None:
    """Render retrieved chunks in a collapsible expander."""
    if not sources:
        return
    with st.expander(f"📎 {len(sources)} retrieved source(s)", expanded=False):
        for i, src in enumerate(sources, 1):
            m     = src.metadata
            score = m.get("rerank_score", 0.0)
            badge_type = {
                "text":  "📝 Text",
                "table": "📊 Table",
                "image": "🖼️ Figure",
            }.get(m.get("type", "text"), "📝 Text")

            st.markdown(
                f"**[Source {i}]** &nbsp;"
                f'<span class="source-badge">{badge_type}</span>'
                f'<span class="source-badge">Score: {score:.3f}</span>'
                f'<span class="source-badge">#{m.get("element_index","?")}</span>',
                unsafe_allow_html=True,
            )
            st.caption(f"File: `{m.get('source', 'N/A')}`")

            snippet = src.page_content
            if len(snippet) > 450:
                snippet = snippet[:450] + " …"
            st.text(snippet)

            if i < len(sources):
                st.divider()
                
# ─────────────────────────────────────────────────────────────────────────────
# Main area
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("## 🔬 Research Paper Multimodal RAG Assistant")
st.caption(
    "Upload an ArXiv PDF → Process → Ask questions with inline citations. "
    "Tables & figures are summarised by a Vision LLM."
)

tab_chat, tab_eval = st.tabs(["💬 Chat", "📊 Evaluation Metrics"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CHAT
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:

    # ── Welcome banner if no paper loaded ─────────────────────────────────────
    pipeline_ready = (
        st.session_state.pipeline is not None
        and st.session_state.pipeline.is_ready
    )

    if not pipeline_ready:
        st.info(
            "👈 **Getting started:**\n\n"
            "1. Enter your **Groq API Key** in the sidebar.\n"
            "2. Upload an **ArXiv PDF** research paper.\n"
            "3. Click **Process Paper** and wait for ingestion to finish.\n"
            "4. Start chatting below!"
        )

    # ── Render existing chat history ──────────────────────────────────────────
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])

            # Show retrieved sources for assistant messages
            if msg["role"] == "assistant" and msg.get("sources"):
                _render_sources(msg["sources"])

    # ── Chat input ────────────────────────────────────────────────────────────
    user_input = st.chat_input(
        "Ask anything about the paper …",
        disabled=not pipeline_ready,
    )

    if user_input:
        if not groq_api_key:
            st.error("Please enter your Groq API Key.")
        else:
            # Display user message immediately
            with st.chat_message("user", avatar="🧑"):
                st.markdown(user_input)
            st.session_state.chat_history.append(
                {"role": "user", "content": user_input}
            )

            # Generate answer
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("🔍 Retrieving & generating …"):
                    result = st.session_state.pipeline.answer(user_input)

                st.markdown(result["answer"])
                _render_sources(result["sources"])

            st.session_state.chat_history.append(
                {
                    "role":    "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                }
            )

    # ── Clear chat button ─────────────────────────────────────────────────────
    if st.session_state.chat_history:
        if st.button("🗑️ Clear conversation", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — EVALUATION METRICS
# ══════════════════════════════════════════════════════════════════════════════
with tab_eval:
    st.markdown("### 📊 RAGAS Evaluation Results")

    if st.session_state.eval_results and "error" not in st.session_state.eval_results:
        results = st.session_state.eval_results

        # ── Helper ────────────────────────────────────────────────────────────
        def _avg(key: str) -> float:
            vals = [v for v in results.get(key, []) if v is not None]
            return sum(vals) / max(len(vals), 1)

        # ── Metric cards ──────────────────────────────────────────────────────
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric(
                "🎯 Faithfulness",
                f"{_avg('faithfulness'):.3f}",
                help="Are the claims in the answer supported by the retrieved context?",
            )
        with c2:
            st.metric(
                "💡 Answer Relevancy",
                f"{_avg('answer_relevancy'):.3f}",
                help="Does the answer actually address the question asked?",
            )
        with c3:
            st.metric(
                "🔍 Context Precision",
                f"{_avg('context_precision'):.3f}",
                help="What fraction of retrieved context was relevant to the question?",
            )

        st.divider()

        # ── Radar / bar chart ─────────────────────────────────────────────────
        st.subheader("Aggregate Scores")
        summary_df = pd.DataFrame(
            {
                "Metric": ["Faithfulness", "Answer Relevancy", "Context Precision"],
                "Score":  [
                    _avg("faithfulness"),
                    _avg("answer_relevancy"),
                    _avg("context_precision"),
                ],
            }
        ).set_index("Metric")
        st.bar_chart(summary_df, color="#58a6ff", height=300)

        # ── Per-question table ────────────────────────────────────────────────
        st.subheader("Per-Question Breakdown")
        df = pd.DataFrame(results)
        # Reorder/rename for clarity
        display_cols = [c for c in ["question", "faithfulness", "answer_relevancy",
                                    "context_precision", "answer"] if c in df.columns]
        st.dataframe(
            df[display_cols].rename(columns={
                "question":          "Question",
                "faithfulness":      "Faithfulness ↑",
                "answer_relevancy":  "Ans. Relevancy ↑",
                "context_precision": "Ctx. Precision ↑",
                "answer":            "Generated Answer",
            }),
            use_container_width=True,
            height=350,
        )

        # ── Download ──────────────────────────────────────────────────────────
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️  Download results (CSV)",
            data=csv_data,
            file_name="ragas_results.csv",
            mime="text/csv",
        )

    else:
        st.info(
            "No evaluation results yet.\n\n"
            "Enter questions in the **RAGAS Evaluation** sidebar panel and "
            "click **▶ Run Evaluation**."
        )

        st.markdown(
            """
---
### 📖 Understanding RAGAS Metrics

| Metric | Range | What it measures |
|---|---|---|
| **Faithfulness** | 0 → 1 | Are all claims in the answer grounded in the retrieved context? High = no hallucinations |
| **Answer Relevancy** | 0 → 1 | Does the answer actually respond to the question? High = focused, on-topic answers |
| **Context Precision** | 0 → 1 | Are the retrieved chunks relevant? High = retriever is precise, not noisy |

*All metrics use your Groq LLM internally — no OpenAI key required.*
"""
        )


# ─────────────────────────────────────────────────────────────────────────────
# Helper — source expander  (defined after tab blocks to avoid forward refs)
# ─────────────────────────────────────────────────────────────────────────────
# def _render_sources(sources) -> None:
#     """Render retrieved chunks in a collapsible expander."""
#     if not sources:
#         return
#     with st.expander(f"📎 {len(sources)} retrieved source(s)", expanded=False):
#         for i, src in enumerate(sources, 1):
#             m     = src.metadata
#             score = m.get("rerank_score", 0.0)
#             badge_type = {
#                 "text":  "📝 Text",
#                 "table": "📊 Table",
#                 "image": "🖼️ Figure",
#             }.get(m.get("type", "text"), "📝 Text")

#             st.markdown(
#                 f"**[Source {i}]** &nbsp;"
#                 f'<span class="source-badge">{badge_type}</span>'
#                 f'<span class="source-badge">Score: {score:.3f}</span>'
#                 f'<span class="source-badge">#{m.get("element_index","?")}</span>',
#                 unsafe_allow_html=True,
#             )
#             st.caption(f"File: `{m.get('source', 'N/A')}`")

#             snippet = src.page_content
#             if len(snippet) > 450:
#                 snippet = snippet[:450] + " …"
#             st.text(snippet)

#             if i < len(sources):
#                 st.divider()
