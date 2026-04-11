import streamlit as st
import os
import time
from dotenv import load_dotenv
from rag.embedder import embed_documents
from rag.retriever import load_vectorstore, search_docs
from rag.llm_chain import get_answer
from utils.pdf_parser import extract_text_from_pdfs

load_dotenv()

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocuMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg-primary: #0a0a0f;
    --bg-secondary: #111118;
    --bg-card: #16161f;
    --bg-input: #1c1c28;
    --accent-blue: #4f8ef7;
    --accent-purple: #9b59f7;
    --accent-cyan: #22d3ee;
    --accent-green: #10b981;
    --text-primary: #f1f1f5;
    --text-secondary: #8888aa;
    --border: #2a2a3a;
    --glow: rgba(79, 142, 247, 0.15);
}

/* Global */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

.stApp { background-color: var(--bg-primary) !important; }

/* Hide default header */
#MainMenu, footer, header { visibility: hidden; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

/* File uploader */
[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 1px dashed var(--accent-blue) !important;
    border-radius: 12px !important;
    padding: 8px !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple)) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 10px 20px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 20px rgba(79, 142, 247, 0.3) !important;
    width: 100% !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(79, 142, 247, 0.5) !important;
}

/* Chat input */
[data-testid="stChatInput"] textarea {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: var(--accent-blue) !important;
    box-shadow: 0 0 0 2px var(--glow) !important;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    margin: 6px 0 !important;
    padding: 4px 8px !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 12px !important;
}
[data-testid="stMetricValue"] { color: var(--accent-cyan) !important; font-family: 'Space Mono', monospace !important; }

/* Expander */
[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--accent-blue); border-radius: 3px; }

/* Spinner */
.stSpinner > div { border-top-color: var(--accent-blue) !important; }
</style>
""", unsafe_allow_html=True)


# ── Header ─────────────────────────────────────────────────────────────────────
def render_header():
    st.markdown("""
    <div style="
        padding: 28px 0 20px 0;
        border-bottom: 1px solid #2a2a3a;
        margin-bottom: 28px;
    ">
        <div style="display:flex; align-items:center; gap:14px;">
            <div style="
                background: linear-gradient(135deg, #4f8ef7, #9b59f7);
                border-radius: 14px;
                width: 52px; height: 52px;
                display: flex; align-items: center; justify-content: center;
                font-size: 26px; box-shadow: 0 4px 20px rgba(79,142,247,0.4);
            ">🧠</div>
            <div>
                <h1 style="margin:0; font-family:'Space Mono',monospace; font-size:28px;
                    background: linear-gradient(135deg, #4f8ef7, #22d3ee);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                    DocuMind AI
                </h1>
                <p style="margin:0; color:#8888aa; font-size:13px; font-family:'DM Sans',sans-serif;">
                    Intelligent Document Intelligence · Powered by RAG
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="padding: 10px 0 20px 0;">
            <h3 style="font-family:'Space Mono',monospace; color:#4f8ef7; margin:0 0 4px 0;">
                📂 Document Hub
            </h3>
            <p style="color:#8888aa; font-size:12px; margin:0;">
                Upload PDFs to begin your session
            </p>
        </div>
        """, unsafe_allow_html=True)

        uploaded_files = st.file_uploader(
            "Drop your PDFs here",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload one or more PDF documents to chat with"
        )

        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            process_btn = st.button("⚡ Process", use_container_width=True)
        with col2:
            clear_btn = st.button("🗑️ Clear", use_container_width=True)

        # Stats
        if st.session_state.get("docs_processed"):
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**📊 Session Stats**")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Docs", st.session_state.get("doc_count", 0))
            with col2:
                st.metric("Chunks", st.session_state.get("chunk_count", 0))
            st.metric("Messages", len(st.session_state.get("messages", [])))

        # Divider
        st.markdown("<br><hr style='border-color:#2a2a3a;'><br>", unsafe_allow_html=True)

        # Sample questions
        st.markdown("**💡 Try asking:**")
        sample_qs = [
            "Summarize this document",
            "What are the key points?",
            "List the main topics",
            "Explain the conclusion"
        ]
        for q in sample_qs:
            if st.button(f"  {q}", key=f"sample_{q}", use_container_width=True):
                st.session_state["prefill_question"] = q

        # About
        st.markdown("<br><hr style='border-color:#2a2a3a;'><br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="color:#8888aa; font-size:11px; line-height:1.6;">
            <b style="color:#4f8ef7;">DocuMind AI</b> uses RAG (Retrieval-Augmented Generation)
            to answer questions from your documents with high accuracy and source citations.<br><br>
            Built with LangChain · FAISS · Groq LLaMA3
        </div>
        """, unsafe_allow_html=True)

        return uploaded_files, process_btn, clear_btn


# ── Welcome Screen ─────────────────────────────────────────────────────────────
def render_welcome():
    st.markdown("""
    <div style="
        display: flex; flex-direction: column; align-items: center;
        justify-content: center; padding: 60px 20px; text-align: center;
    ">
        <div style="
            font-size: 72px; margin-bottom: 24px;
            filter: drop-shadow(0 0 30px rgba(79,142,247,0.5));
        ">🧠</div>
        <h2 style="
            font-family: 'Space Mono', monospace;
            background: linear-gradient(135deg, #4f8ef7, #22d3ee, #9b59f7);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            font-size: 32px; margin: 0 0 12px 0;
        ">Chat With Your Documents</h2>
        <p style="color: #8888aa; font-size: 16px; max-width: 480px; line-height: 1.7; margin: 0 0 40px 0;">
            Upload your PDFs in the sidebar, click <b style="color:#4f8ef7;">Process</b>,
            and start asking questions. Get accurate, cited answers instantly.
        </p>
        <div style="display: flex; gap: 20px; flex-wrap: wrap; justify-content: center;">
    """ + "".join([f"""
            <div style="
                background: #16161f; border: 1px solid #2a2a3a;
                border-radius: 14px; padding: 20px 24px; width: 160px;
                transition: all 0.3s ease;
            ">
                <div style="font-size: 28px; margin-bottom: 10px;">{icon}</div>
                <div style="font-size: 13px; color: #8888aa; font-weight: 500;">{text}</div>
            </div>
    """ for icon, text in [
        ("📄", "Upload PDFs"),
        ("⚡", "Instant Processing"),
        ("💬", "Smart Q&A"),
        ("📎", "Source Citations"),
    ]]) + """
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Chat Message ───────────────────────────────────────────────────────────────
def render_chat_message(role, content, sources=None):
    with st.chat_message(role, avatar="🧠" if role == "assistant" else "👤"):
        st.markdown(content)
        if sources and role == "assistant":
            with st.expander("📎 View Sources", expanded=False):
                for i, src in enumerate(sources, 1):
                    st.markdown(f"""
                    <div style="
                        background: #1c1c28; border-left: 3px solid #4f8ef7;
                        border-radius: 8px; padding: 10px 14px; margin: 6px 0;
                        font-size: 13px; color: #c0c0d0; line-height: 1.6;
                    ">
                        <span style="color:#4f8ef7; font-weight:600;">Source {i}</span>
                        — Page {src.metadata.get('page', '?')} · {src.metadata.get('source', 'document')}<br>
                        <span style="color:#8888aa;">"{src.page_content[:220]}..."</span>
                    </div>
                    """, unsafe_allow_html=True)


# ── Processing Status ──────────────────────────────────────────────────────────
def show_processing_status(message, status="info"):
    colors = {"info": "#4f8ef7", "success": "#10b981", "error": "#ef4444"}
    icons = {"info": "⚡", "success": "✅", "error": "❌"}
    color = colors.get(status, "#4f8ef7")
    icon = icons.get(status, "⚡")
    st.markdown(f"""
    <div style="
        background: #16161f; border: 1px solid {color}33;
        border-left: 4px solid {color}; border-radius: 10px;
        padding: 14px 18px; margin: 10px 0;
        display: flex; align-items: center; gap: 10px;
    ">
        <span style="font-size: 18px;">{icon}</span>
        <span style="color: {color}; font-weight: 500; font-size: 14px;">{message}</span>
    </div>
    """, unsafe_allow_html=True)


# ── Session State Init ─────────────────────────────────────────────────────────
def init_session():
    defaults = {
        "messages": [],
        "docs_processed": False,
        "vectorstore": None,
        "doc_count": 0,
        "chunk_count": 0,
        "prefill_question": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    init_session()
    render_header()

    uploaded_files, process_btn, clear_btn = render_sidebar()

    # Clear chat
    if clear_btn:
        st.session_state["messages"] = []
        st.session_state["docs_processed"] = False
        st.session_state["vectorstore"] = None
        st.session_state["doc_count"] = 0
        st.session_state["chunk_count"] = 0
        st.rerun()

    # Process documents
    if process_btn:
        if not uploaded_files:
            st.sidebar.warning("⚠️ Please upload at least one PDF first.")
        else:
            with st.sidebar:
                with st.spinner("Processing documents..."):
                    try:
                        texts, metadatas = extract_text_from_pdfs(uploaded_files)
                        vs, chunks = embed_documents(texts, metadatas)
                        st.session_state["vectorstore"] = vs
                        st.session_state["docs_processed"] = True
                        st.session_state["doc_count"] = len(uploaded_files)
                        st.session_state["chunk_count"] = chunks
                        st.session_state["messages"] = []
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

    # Main chat area
    if not st.session_state["docs_processed"]:
        render_welcome()
    else:
        # Success banner
        show_processing_status(
            f"✅  {st.session_state['doc_count']} document(s) ready · "
            f"{st.session_state['chunk_count']} chunks indexed · Ask anything below",
            "success"
        )

        # Chat history
        chat_container = st.container()
        with chat_container:
            if not st.session_state["messages"]:
                st.markdown("""
                <div style="text-align:center; padding: 40px 0 20px 0; color:#8888aa;">
                    <div style="font-size:36px; margin-bottom:12px;">💬</div>
                    <p style="font-size:15px;">Documents are ready! Start asking questions below.</p>
                </div>
                """, unsafe_allow_html=True)

            for msg in st.session_state["messages"]:
                render_chat_message(
                    msg["role"],
                    msg["content"],
                    msg.get("sources")
                )

        # Prefill from sidebar sample buttons
        prefill = st.session_state.pop("prefill_question", None)

        # Chat input
        user_input = st.chat_input("Ask anything about your documents...")

        if prefill and not user_input:
            user_input = prefill

        if user_input:
            # Add user message
            st.session_state["messages"].append({"role": "user", "content": user_input})
            render_chat_message("user", user_input)

            # Get answer
            with st.chat_message("assistant", avatar="🧠"):
                with st.spinner("Searching documents and generating answer..."):
                    try:
                        docs = search_docs(st.session_state["vectorstore"], user_input)
                        answer = get_answer(user_input, docs)

                        # Typing effect
                        placeholder = st.empty()
                        displayed = ""
                        for char in answer:
                            displayed += char
                            placeholder.markdown(displayed + "▌")
                            time.sleep(0.008)
                        placeholder.markdown(answer)

                        # Sources
                        if docs:
                            with st.expander("📎 View Sources", expanded=False):
                                for i, src in enumerate(docs, 1):
                                    st.markdown(f"""
                                    <div style="
                                        background:#1c1c28; border-left:3px solid #4f8ef7;
                                        border-radius:8px; padding:10px 14px; margin:6px 0;
                                        font-size:13px; color:#c0c0d0; line-height:1.6;
                                    ">
                                        <span style="color:#4f8ef7; font-weight:600;">Source {i}</span>
                                        — Page {src.metadata.get('page','?')}<br>
                                        <span style="color:#8888aa;">"{src.page_content[:220]}..."</span>
                                    </div>
                                    """, unsafe_allow_html=True)

                        st.session_state["messages"].append({
                            "role": "assistant",
                            "content": answer,
                            "sources": docs
                        })

                    except Exception as e:
                        err_msg = f"⚠️ Error: {str(e)}"
                        st.error(err_msg)
                        st.session_state["messages"].append({
                            "role": "assistant",
                            "content": err_msg
                        })


if __name__ == "__main__":
    main()