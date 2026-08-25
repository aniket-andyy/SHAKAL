"""
SHAKAL - STUDY MADAD
A RAG-based Study AI Assistant.

Streamlit frontend only. All RAG/LLM/embedding logic lives in backend/
(refactored from the original main.py / database.py / embedding.py)
and is imported here, not duplicated.
"""

import os
import tempfile

import streamlit as st

from backend.loaders import process_source, SourceProcessingError
from backend.rag import (
    generate_response,
    MODE_SOURCE_LLM,
    MODE_SOURCE_ONLY,
    MODE_LLM_ONLY,
)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="SHAKAL — Study Madad",
    page_icon="🗿",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ============================================================
# STYLES
# ============================================================

def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

        :root {
            --bg-main: #0a0a0c;
            --bg-card: #131316;
            --bg-input: #1a1a1e;
            --border-subtle: #2a2a30;
            --border-highlight: #3d3d46;
            --accent: #6ee7b7;
            --accent-dim: #34a37a;
            --text-primary: #f2f2f5;
            --text-secondary: #9a9aa5;
            --text-muted: #5f5f6a;
            --danger: #f28b82;
        }

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header[data-testid="stHeader"] {background: transparent;}
        .stDeployButton {display: none;}

        .stApp {
            background: var(--bg-main);
            color: var(--text-primary);
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 760px;
        }

        /* ---------- Header ---------- */
        .shakal-header {
            text-align: center;
            padding: 1.5rem 0 2rem 0;
        }
        .shakal-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 2.6rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            color: var(--text-primary);
            margin-bottom: -0.2rem;
        }
        .shakal-subtitle {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 0.95rem;
            font-weight: 500;
            letter-spacing: 0.35em;
            color: var(--accent);
            text-transform: uppercase;
            margin-bottom: 0.6rem;
        }
        .shakal-tagline {
            font-size: 0.9rem;
            color: var(--text-secondary);
            margin-bottom: 0.9rem;
        }
        .shakal-model-tag {
            display: inline-block;
            font-size: 0.78rem;
            color: var(--text-secondary);
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            padding: 0.3rem 0.8rem;
            border-radius: 999px;
            margin-bottom: 0.7rem;
        }
        .shakal-credit {
            font-size: 0.8rem;
            color: var(--text-muted);
        }
        .shakal-credit a {
            color: var(--text-secondary);
            text-decoration: none;
            border-bottom: 1px solid var(--border-highlight);
            padding-bottom: 1px;
        }
        .shakal-credit a:hover {
            color: var(--accent);
            border-bottom-color: var(--accent);
        }

        /* ---------- Section headings ---------- */
        .section-heading {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.25rem;
            font-weight: 600;
            color: var(--text-primary);
            text-align: center;
            margin-bottom: 0.15rem;
        }
        .section-subheading {
            font-size: 0.88rem;
            color: var(--text-secondary);
            text-align: center;
            margin-bottom: 1.3rem;
        }

        /* ---------- Cards ---------- */
        .shakal-card {
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: 14px;
            padding: 1.1rem 1.3rem;
            margin-bottom: 1rem;
        }

        .source-ready-card {
            background: linear-gradient(135deg, rgba(110,231,183,0.07), rgba(110,231,183,0.02));
            border: 1px solid rgba(110,231,183,0.3);
            border-radius: 14px;
            padding: 0.9rem 1.2rem;
            margin-bottom: 1.2rem;
        }
        .source-ready-label {
            font-size: 0.68rem;
            letter-spacing: 0.15em;
            color: var(--accent);
            font-weight: 600;
            margin-bottom: 0.25rem;
        }
        .source-ready-value {
            font-size: 0.92rem;
            color: var(--text-primary);
        }

        /* ---------- Status text ---------- */
        .status-line {
            font-size: 0.85rem;
            color: var(--text-secondary);
            text-align: center;
            padding: 0.4rem 0;
        }

        .processing-block {
            text-align: center;
            padding: 0.6rem 0 0.2rem 0;
        }
        .processing-label {
            font-size: 0.85rem;
            color: var(--text-secondary);
            font-weight: 500;
        }
        .processing-note {
            font-size: 0.75rem;
            color: var(--text-muted);
            margin-top: 0.15rem;
            font-style: italic;
        }

        /* ---------- Chat bubbles ---------- */
        .chat-row {
            display: flex;
            margin-bottom: 0.85rem;
        }
        .chat-row.user { justify-content: flex-end; }
        .chat-row.ai { justify-content: flex-start; }

        .chat-bubble {
            max-width: 82%;
            padding: 0.7rem 1rem;
            border-radius: 14px;
            font-size: 0.92rem;
            line-height: 1.5;
        }
        .chat-bubble.user {
            background: var(--bg-input);
            border: 1px solid var(--border-highlight);
            border-bottom-right-radius: 4px;
        }
        .chat-bubble.ai {
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-bottom-left-radius: 4px;
        }
        .chat-label {
            font-size: 0.65rem;
            letter-spacing: 0.12em;
            font-weight: 600;
            margin-bottom: 0.3rem;
            display: block;
        }
        .chat-label.user { color: var(--text-muted); text-align: right; }
        .chat-label.ai { color: var(--accent); }

        /* ---------- Buttons ---------- */
        .stButton > button {
            background: var(--text-primary);
            color: #0a0a0c;
            border: none;
            border-radius: 10px;
            font-weight: 600;
            font-size: 0.88rem;
            padding: 0.5rem 1.2rem;
            transition: opacity 0.15s ease;
        }
        .stButton > button:hover {
            opacity: 0.85;
            color: #0a0a0c;
        }

        /* Inputs */
        .stTextInput input, .stTextArea textarea {
            background: var(--bg-input) !important;
            border: 1px solid var(--border-subtle) !important;
            color: var(--text-primary) !important;
            border-radius: 10px !important;
        }
        .stTextInput input:focus, .stTextArea textarea:focus {
            border-color: var(--accent) !important;
        }

        [data-testid="stFileUploaderDropzone"] {
            background: var(--bg-input);
            border: 1px dashed var(--border-highlight);
            border-radius: 12px;
        }

        hr {
            border-color: var(--border-subtle) !important;
            margin: 1.6rem 0 !important;
        }

        /* Segmented control theming */
        div[data-testid="stSegmentedControl"] label {
            font-size: 0.82rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# SESSION STATE
# ============================================================

def init_session_state():
    defaults = {
        "mode": "Source + LLM/AI",
        "source_type": None,
        "source_ready": False,
        "source_label": None,
        "source_icon": None,
        "chat_history": [],
        "processing": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ============================================================
# RENDER FUNCTIONS
# ============================================================

def render_header():
    st.markdown(
        """
        <div class="shakal-header">
            <div class="shakal-title">SHAKAL</div>
            <div class="shakal-subtitle">Study Madad</div>
            <div class="shakal-tagline">Your AI Study Assistant</div>
            <div class="shakal-model-tag">Model: Mistral Small 2506</div>
            <div class="shakal-credit">
                Developed by Aniket Sharma &nbsp;·&nbsp;
                <a href="https://www.linkedin.com/in/aniket-sharma-42a700418?utm_source=share_via&utm_content=member_android" target="_blank">LinkedIn</a>
                &nbsp;|&nbsp;
                <a href="https://github.com/aniket-andyy" target="_blank">GitHub</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


SOURCE_OPTIONS = {
    "PDF": {"icon": "📄", "type": "pdf"},
    "Website Link": {"icon": "🌐", "type": "website"},
    "YouTube Video": {"icon": "🎥", "type": "youtube"},
    "Image": {"icon": "🖼️", "type": "image"},
}


def render_source_ready_card():
    st.markdown(
        f"""
        <div class="source-ready-card">
            <div class="source-ready-label">SOURCE READY</div>
            <div class="source-ready-value">{st.session_state.source_icon} {st.session_state.source_label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_source_section():
    st.markdown('<div class="section-heading">Upload Your Source</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subheading">Give SHAKAL your study material and start learning.</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.source_ready:
        render_source_ready_card()
        if st.button("Upload a different source", use_container_width=False):
            st.session_state.source_ready = False
            st.session_state.source_type = None
            st.session_state.source_label = None
            st.session_state.source_icon = None
            st.rerun()
        return

    labels = list(SOURCE_OPTIONS.keys())
    icons = [SOURCE_OPTIONS[label]["icon"] for label in labels]
    display_labels = [f"{icon}  {label}" for icon, label in zip(icons, labels)]

    selected_display = st.radio(
        "Source type",
        options=display_labels,
        horizontal=True,
        label_visibility="collapsed",
        key="source_type_selector",
    )
    selected_label = labels[display_labels.index(selected_display)]
    selected_type = SOURCE_OPTIONS[selected_label]["type"]
    selected_icon = SOURCE_OPTIONS[selected_label]["icon"]

    st.markdown("<br>", unsafe_allow_html=True)

    # --- PDF ---
    if selected_type == "pdf":
        uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
        if uploaded_file is not None:
            size_kb = uploaded_file.size / 1024
            st.markdown(
                f'<div class="status-line">{uploaded_file.name} · {size_kb:.1f} KB · ready to process</div>',
                unsafe_allow_html=True,
            )
            if st.button("Process Source", key="process_pdf"):
                handle_process_source("pdf", uploaded_file, selected_icon)

    # --- WEBSITE ---
    elif selected_type == "website":
        url = st.text_input("Paste Website URL", placeholder="https://example.com/article")
        if st.button("Process Source", key="process_website"):
            if url.strip():
                handle_process_source("website", url.strip(), selected_icon)
            else:
                st.markdown('<div class="status-line">Please enter a URL first.</div>', unsafe_allow_html=True)

    # --- YOUTUBE ---
    elif selected_type == "youtube":
        url = st.text_input("Paste YouTube Video URL", placeholder="https://youtube.com/watch?v=...")
        if st.button("Process Source", key="process_youtube"):
            if url.strip():
                handle_process_source("youtube", url.strip(), selected_icon)
            else:
                st.markdown('<div class="status-line">Please enter a URL first.</div>', unsafe_allow_html=True)

    # --- IMAGE ---
    elif selected_type == "image":
        uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg", "webp"])
        if uploaded_file is not None:
            size_kb = uploaded_file.size / 1024
            st.markdown(
                f'<div class="status-line">{uploaded_file.name} · {size_kb:.1f} KB · ready to process</div>',
                unsafe_allow_html=True,
            )
            if st.button("Process Source", key="process_image"):
                handle_process_source("image", uploaded_file, selected_icon)


def handle_process_source(source_type, value, icon):
    """Runs process_source() with staged status messages, then updates session state."""
    status = st.empty()

    stages = ["Processing source...", "Extracting content...", "Creating embeddings...", "Updating knowledge base..."]
    for stage in stages:
        status.markdown(f'<div class="status-line">{stage}</div>', unsafe_allow_html=True)

    try:
        # File uploads (pdf/image) need to be written to a temp path for the
        # existing loaders (PyPDFLoader / OCR request) which expect a file path.
        if source_type in ("pdf", "image"):
            suffix = os.path.splitext(value.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(value.getbuffer())
                tmp_path = tmp.name

            label_hint = value.name
            result = process_source(source_type, tmp_path)
            result["label"] = label_hint  # prefer original filename over temp path
            os.unlink(tmp_path)
        else:
            result = process_source(source_type, value)

        status.markdown('<div class="status-line">Source ready.</div>', unsafe_allow_html=True)

        st.session_state.source_ready = True
        st.session_state.source_type = source_type
        st.session_state.source_label = result["label"]
        st.session_state.source_icon = icon
        st.rerun()

    except SourceProcessingError as e:
        status.markdown(f'<div class="status-line">{e}</div>', unsafe_allow_html=True)
    except Exception:
        status.markdown(
            '<div class="status-line">Unable to process this source. Please check the file or URL and try again.</div>',
            unsafe_allow_html=True,
        )


MODE_LABELS = {
    "Source + LLM/AI": {
        "value": MODE_SOURCE_LLM,
        "desc": "Uses your source as the primary knowledge and AI knowledge when needed.",
    },
    "Only My Source": {
        "value": MODE_SOURCE_ONLY,
        "desc": "Answers strictly from your uploaded source.",
    },
    "Only LLM/AI": {
        "value": MODE_LLM_ONLY,
        "desc": "Answers using the AI's general knowledge without your source.",
    },
}


def render_mode_selector():
    st.markdown('<div class="section-heading">AI Working Mode</div>', unsafe_allow_html=True)

    options = list(MODE_LABELS.keys())

    try:
        selected = st.segmented_control(
            "Mode",
            options=options,
            default=st.session_state.mode,
            label_visibility="collapsed",
        )
        if selected:
            st.session_state.mode = selected
    except AttributeError:
        # Fallback for older Streamlit versions without segmented_control
        selected = st.radio(
            "Mode",
            options=options,
            index=options.index(st.session_state.mode),
            horizontal=True,
            label_visibility="collapsed",
        )
        st.session_state.mode = selected

    st.markdown(
        f'<div class="section-subheading">{MODE_LABELS[st.session_state.mode]["desc"]}</div>',
        unsafe_allow_html=True,
    )


def render_chat():
    st.markdown('<div class="section-heading">Ask SHAKAL</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    for entry in st.session_state.chat_history:
        role = entry["role"]
        text = entry["content"]
        if role == "user":
            st.markdown(
                f"""
                <div class="chat-row user">
                    <div>
                        <span class="chat-label user">YOU</span>
                        <div class="chat-bubble user">{text}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="chat-row ai">
                    <div>
                        <span class="chat-label ai">SHAKAL</span>
                        <div class="chat-bubble ai">{text}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if st.session_state.processing:
        st.markdown(
            """
            <div class="processing-block">
                <div class="processing-label">Processing</div>
                <div class="processing-note">[ aniket bhai ki taraf se hello! :) ]</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        with col1:
            query = st.text_input(
                "Ask SHAKAL anything about your studies...",
                placeholder="Ask SHAKAL anything about your studies...",
                label_visibility="collapsed",
            )
        with col2:
            submitted = st.form_submit_button("Send", use_container_width=True)

    if submitted and query.strip():
        st.session_state.chat_history.append({"role": "user", "content": query.strip()})
        st.session_state.processing = True
        st.rerun()


def handle_pending_response():
    """If the last message is an unanswered user query, generate a response now."""
    if not st.session_state.processing:
        return

    history = st.session_state.chat_history
    if not history or history[-1]["role"] != "user":
        st.session_state.processing = False
        return

    query = history[-1]["content"]
    mode_value = MODE_LABELS[st.session_state.mode]["value"]

    try:
        response_text = generate_response(query, mode_value)
    except Exception:
        response_text = "Something went wrong while generating a response. Please try again."

    st.session_state.chat_history.append({"role": "ai", "content": response_text})
    st.session_state.processing = False
    st.rerun()


# ============================================================
# MAIN
# ============================================================

def main():
    inject_css()
    init_session_state()

    render_header()
    st.markdown("<hr>", unsafe_allow_html=True)

    render_source_section()
    st.markdown("<hr>", unsafe_allow_html=True)

    render_mode_selector()
    st.markdown("<hr>", unsafe_allow_html=True)

    render_chat()

    # Runs after the "Processing" indicator has already been rendered above,
    # so the user sees it before the (blocking) LLM call happens.
    handle_pending_response()


if __name__ == "__main__":
    main()
