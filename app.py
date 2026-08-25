import os
import time
import tempfile
import html

import streamlit as st

from database import (
    load_pdf,
    load_webpage,
    load_youtube,
    load_image,
    add_to_chroma
)

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(
    page_title="SHAKAL - STUDY MADAD [TERMINAL]",
    page_icon="▶",
    layout="wide"
)

# ==========================================
# AI QUOTES & FACTS
# ==========================================
AI_QUOTES = [
    "“The best way to predict the future is to invent it.” — Alan Kay",
    "“Intelligence is the ability to adapt to change.” — Stephen Hawking",
    "“AI is probably the most important thing humanity has ever worked on.” — Sundar Pichai",
    "“Machine intelligence is the last invention humanity will ever need to make.” — Nick Bostrom",
    "⚡ The term 'Artificial Intelligence' was coined in 1956 at Dartmouth.",
    "⚡ The first chatbot, ELIZA, was built at MIT in 1966.",
    "⚡ Neural networks are inspired by human brain neurons.",
    "⚡ Over 90% of the world's data was created in just the last few years.",
    "⚡ RAG = how SHAKAL reads your sources!",
    "“Whether machines think is like asking if submarines can swim.” — Dijkstra",
    "⚡ Vector embeddings turn text into numbers — AI measures meaning like distance.",
    "“Learning is not attained by chance, it must be sought for with ardor.” — Abigail Adams",
]

if "fact_index" not in st.session_state:
    st.session_state.fact_index = 0

def next_fact():
    fact = AI_QUOTES[st.session_state.fact_index % len(AI_QUOTES)]
    st.session_state.fact_index += 1
    return fact

# ==========================================
# ASCII LOGO (flush-left, no indentation!)
# ==========================================
ASCII_LOGO = """<pre class="ascii-logo">  ____  _   _    _    _  __    _    _
 / ___|| | | |  / \\  | |/ /   / \\  | |
 \\___ \\| |_| | / _ \\ | ' /   / _ \\ | |
  ___) |  _  |/ ___ \\| . \\  / ___ \\| |___
 |____/|_| |_/_/   \\_\\_|\\_\\/_/   \\_\\_____|
</pre>"""

# ==========================================
# TERMINAL CSS
# ==========================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=JetBrains+Mono:wght@400;500;700&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'JetBrains Mono', 'Courier New', monospace !important;
    background: #0a0e0a !important;
    color: #33ff33 !important;
}

/* CRT Scanlines */
.stApp::before {
    content: "";
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: repeating-linear-gradient(
        0deg,
        rgba(0,0,0,0) 0px, rgba(0,0,0,0) 2px,
        rgba(0,0,0,0.15) 3px, rgba(0,0,0,0.15) 3px
    );
    pointer-events: none;
    z-index: 9999;
}

@keyframes flicker { 0%,100% {opacity:.98;} 50% {opacity:1;} }
.stApp { animation: flicker 4s infinite; }

#MainMenu, footer, header, .stDeployButton, [data-testid="stHeader"] {
    visibility: hidden !important; height: 0 !important;
}

.block-container {
    max-width: 1000px;
    padding: 2rem 1rem 4rem 1rem !important;
}

pre { background: transparent !important; }

/* ===== LOGO ===== */
.ascii-logo {
    font-family: 'Share Tech Mono', monospace;
    color: #33ff33;
    text-shadow: 0 0 5px #33ff33, 0 0 10px #33ff33, 0 0 20px #00ff00;
    font-size: clamp(7px, 2.2vw, 16px);
    line-height: 1.15;
    text-align: center;
    margin: 20px 0 10px 0;
    white-space: pre;
    overflow-x: hidden;
}

.ascii-subtitle {
    font-family: 'Share Tech Mono', monospace;
    color: #ffcc00;
    text-shadow: 0 0 8px #ffcc00;
    text-align: center;
    font-size: clamp(12px, 1.6vw, 18px);
    letter-spacing: 8px;
    margin: 8px 0;
}

.ascii-tagline {
    text-align: center;
    color: #33ff33;
    font-size: 14px;
    margin: 4px 0 24px 0;
    opacity: 0.7;
}
.ascii-tagline::before { content: ">> "; color: #ffcc00; }
.ascii-tagline::after  { content: " <<"; color: #ffcc00; }

.ascii-meta {
    font-family: 'Share Tech Mono', monospace;
    text-align: center;
    color: #1a7a1a;
    font-size: 12px;
    margin: 4px 0;
}
.ascii-meta b { color: #33ff33; }
.ascii-meta a { color: #00ffff; text-decoration: none; }

.ascii-hr {
    color: #1a7a1a;
    text-align: center;
    font-size: 12px;
    overflow: hidden;
    margin: 20px 0;
    white-space: nowrap;
}

/* ===== SECTIONS ===== */
.ascii-section {
    font-family: 'Share Tech Mono', monospace;
    color: #ffcc00;
    text-shadow: 0 0 8px #ffcc00;
    font-size: clamp(14px, 1.8vw, 18px);
    margin: 40px 0 12px 0;
    letter-spacing: 2px;
}
.ascii-section::before { content: "╭─ "; }
.ascii-section::after  { content: " ────────────╮"; opacity: 0.5; }

.ascii-subsection {
    color: #1a7a1a;
    font-size: 13px;
    margin-bottom: 20px;
    padding-left: 24px;
}
.ascii-subsection::before { content: "│  "; color: #ffcc00; }

.ascii-note {
    color: #ffcc00;
    font-size: 13px;
    padding: 10px 16px;
    border: 1px dashed #ffcc00;
    margin-bottom: 20px;
    background: rgba(255, 204, 0, 0.03);
}
.ascii-note::before { content: "$ "; color: #33ff33; font-weight: 700; }

/* ===== BUTTONS (brackets are literal in labels now) ===== */
.stButton > button {
    font-family: 'Share Tech Mono', monospace !important;
    background: transparent !important;
    color: #33ff33 !important;
    border: 1px solid #33ff33 !important;
    border-radius: 0 !important;
    padding: 8px 16px !important;
    font-size: 13px !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    transition: all 0.15s linear !important;
    box-shadow: none !important;
    font-weight: 400 !important;
    backdrop-filter: none !important;
}

.stButton > button:hover {
    background: #33ff33 !important;
    color: #0a0e0a !important;
    box-shadow: 0 0 15px #33ff33 !important;
    transform: none !important;
}

.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"],
.stButton > button[data-testid="baseButton-primary"] {
    background: #ffcc00 !important;
    color: #0a0e0a !important;
    border: 1px solid #ffcc00 !important;
    font-weight: 700 !important;
    box-shadow: 0 0 12px #ffcc00 !important;
}
.stButton > button[kind="primary"]:hover {
    background: #ff9900 !important;
    box-shadow: 0 0 20px #ff9900 !important;
}

/* ===== PERSONALITY PILLS (scoped ONLY to radiogroup options) ===== */
[data-testid="stRadio"] [role="radiogroup"] {
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: wrap !important;
    gap: 10px !important;
    justify-content: flex-start !important;
    margin-left: 24px !important;
}

[data-testid="stRadio"] [role="radiogroup"] label {
    font-family: 'Share Tech Mono', monospace !important;
    background: transparent !important;
    color: #33ff33 !important;
    border: 1px solid #1a7a1a !important;
    border-radius: 0 !important;
    padding: 8px 14px !important;
    margin: 0 !important;
    font-size: 13px !important;
    letter-spacing: 1px !important;
    white-space: nowrap !important;
    transition: all 0.15s linear !important;
    cursor: pointer !important;
    text-transform: uppercase !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    box-shadow: none !important;
    backdrop-filter: none !important;
}

[data-testid="stRadio"] [role="radiogroup"] label::before { content: "[ "; color: #1a7a1a; }
[data-testid="stRadio"] [role="radiogroup"] label::after  { content: " ]"; color: #1a7a1a; }

[data-testid="stRadio"] [role="radiogroup"] label:hover {
    border-color: #33ff33 !important;
    text-shadow: 0 0 6px #33ff33 !important;
    box-shadow: 0 0 12px #33ff33 !important;
}

[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) {
    background: #ffcc00 !important;
    color: #0a0e0a !important;
    border-color: #ffcc00 !important;
    font-weight: 700 !important;
    box-shadow: 0 0 15px #ffcc00 !important;
    text-shadow: none !important;
}
[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked)::before,
[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked)::after {
    color: #0a0e0a !important;
}

/* Hide only the radio DOT, never the label */
[data-testid="stRadio"] [role="radiogroup"] svg,
[data-testid="stRadio"] [role="radiogroup"] input[type="radio"],
[data-testid="stRadio"] [role="radiogroup"] [data-baseweb="radio"] {
    display: none !important;
    width: 0 !important; height: 0 !important; margin: 0 !important;
}
[data-testid="stRadio"] [role="radiogroup"] label > div {
    padding: 0 !important; margin: 0 !important;
}

/* ===== UPLOADERS / TEXTAREAS ===== */
[data-testid="stFileUploader"] {
    border: 1px dashed #33ff33 !important;
    background: rgba(51, 255, 51, 0.02) !important;
    border-radius: 0 !important;
}
[data-testid="stFileUploader"] * { color: #33ff33 !important; }
[data-testid="stFileUploaderDropzone"] {
    border: 1px dashed #33ff33 !important;
    background: transparent !important;
}
[data-testid="stFileUploader"] button {
    background: transparent !important;
    color: #ffcc00 !important;
    border: 1px solid #ffcc00 !important;
    border-radius: 0 !important;
}

textarea, [data-testid="stTextArea"] textarea {
    font-family: 'Share Tech Mono', monospace !important;
    background: #0a0e0a !important;
    color: #33ff33 !important;
    border: 1px solid #1a7a1a !important;
    border-radius: 0 !important;
    caret-color: #33ff33 !important;
}
textarea:focus { border-color: #33ff33 !important; box-shadow: 0 0 10px #33ff33 !important; }
textarea::placeholder { color: #1a7a1a !important; }

/* ===== PROCESSING BOX ===== */
.processing {
    font-family: 'Share Tech Mono', monospace;
    color: #33ff33;
    background: rgba(51, 255, 51, 0.03);
    border: 1px solid #33ff33;
    padding: 16px 20px;
    margin: 16px 0;
    font-size: 13px;
    line-height: 1.5;
    text-shadow: 0 0 4px #33ff33;
    box-shadow: 0 0 15px rgba(51, 255, 51, 0.2);
}
.processing-main {
    font-size: 15px;
    color: #ffcc00;
    text-shadow: 0 0 8px #ffcc00;
    font-weight: 700;
}
.processing-main::before { content: "▶ "; }
@keyframes blink-cursor { 0%,49% {opacity:1;} 50%,100% {opacity:0;} }
.processing-main::after { content: "█"; animation: blink-cursor 1s infinite; color: #33ff33; }
.processing-message { color: #00ffff; margin-top: 6px; }
.processing-message::before { content: "$ "; color: #33ff33; }
.processing-fact { color: #33ff33; margin-top: 8px; padding-top: 8px; border-top: 1px dashed #1a7a1a; }
.processing-fact::before { content: "ℹ "; color: #ffcc00; }

/* ===== CHAT ===== */
.stChatMessage, [data-testid="stChatMessage"] {
    background: transparent !important;
    border: 1px solid #1a7a1a !important;
    border-radius: 0 !important;
    margin: 12px 0 !important;
    padding: 12px 16px !important;
}
[data-testid="stChatMessage"] p { color: #33ff33 !important; }
[data-testid="stChatMessage"] pre { background: #0f140f !important; color: #33ff33 !important; }

.stChatInput textarea, [data-testid="stChatInputTextArea"] {
    font-family: 'Share Tech Mono', monospace !important;
    background: #0a0e0a !important;
    color: #33ff33 !important;
    border: 1px solid #33ff33 !important;
    border-radius: 0 !important;
    padding: 12px 16px !important;
    caret-color: #33ff33 !important;
    min-height: 50px !important;
    line-height: 24px !important;
}
.stChatInput textarea::placeholder, [data-testid="stChatInputTextArea"]::placeholder {
    color: #1a7a1a !important;
}
.stChatInput textarea:focus, [data-testid="stChatInputTextArea"]:focus {
    border-color: #ffcc00 !important;
    box-shadow: 0 0 12px #ffcc00 !important;
}

.source-chip {
    font-family: 'Share Tech Mono', monospace;
    color: #33ff33;
    border: 1px solid #1a7a1a;
    padding: 6px 12px;
    margin: 4px;
    display: inline-block;
    font-size: 12px;
    text-align: center;
    width: 100%;
}

.personality-info {
    text-align: center;
    color: #1a7a1a;
    font-size: 12px;
    margin-top: 14px;
}

[data-testid="stCaption"] { color: #1a7a1a !important; font-family: 'Share Tech Mono', monospace !important; }
[data-testid="stAlert"] {
    background: transparent !important;
    border: 1px solid #ffcc00 !important;
    border-radius: 0 !important;
    color: #ffcc00 !important;
    font-family: 'Share Tech Mono', monospace !important;
}
[data-testid="stAlert"] * { color: #ffcc00 !important; }

.stProgress > div > div { background: #1a7a1a !important; border-radius: 0 !important; }
.stProgress > div > div > div > div {
    background: #33ff33 !important;
    box-shadow: 0 0 10px #33ff33 !important;
    border-radius: 0 !important;
}

.source-ready {
    font-family: 'Share Tech Mono', monospace;
    color: #33ff33;
    border: 1px solid #00ff00;
    padding: 14px 20px;
    margin: 16px 0;
    background: rgba(0, 255, 0, 0.03);
    box-shadow: 0 0 15px rgba(0, 255, 0, 0.2);
}
.source-ready-title {
    color: #00ff00;
    text-shadow: 0 0 8px #00ff00;
    font-size: 15px;
    font-weight: 700;
}
.source-ready-title::before { content: "✓ "; }
.source-ready-text { color: #1a7a1a; margin-top: 6px; font-size: 12px; }
.source-ready-text::before { content: "> "; color: #33ff33; }

::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: #0a0e0a; }
::-webkit-scrollbar-thumb { background: #1a7a1a; }
::selection { background: #33ff33; color: #0a0e0a; }
a { color: #00ffff !important; text-decoration: none !important; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# STATE
# ==========================================
if "sources" not in st.session_state:
    st.session_state.sources = []
if "source_ready" not in st.session_state:
    st.session_state.source_ready = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "personality" not in st.session_state:
    st.session_state.personality = "Normal"

# ==========================================
# HEADER (flush-left HTML!)
# ==========================================
st.markdown(ASCII_LOGO, unsafe_allow_html=True)

st.markdown("""
<div class="ascii-subtitle">&gt; S T U D Y &nbsp; M A D A D &lt;</div>
<div class="ascii-tagline">Your AI Study Assistant</div>
<div class="ascii-meta">MODEL: <b>mistral-small-2506</b></div>
<div class="ascii-meta">DEV&nbsp; : <b>Aniket Sharma</b></div>
<div class="ascii-meta"><a href="https://www.linkedin.com/in/aniket-sharma-42a700418" target="_blank">[LinkedIn]</a> &nbsp;·&nbsp; <a href="https://github.com/aniket-andyy" target="_blank">[GitHub]</a></div>
""", unsafe_allow_html=True)

st.markdown('<pre class="ascii-hr">════════════════════════════════════════════════════════════</pre>', unsafe_allow_html=True)

# ==========================================
# UPLOAD SOURCES
# ==========================================
st.markdown('<div class="ascii-section">UPLOAD_SOURCES</div>', unsafe_allow_html=True)
st.markdown('<div class="ascii-subsection">Feed SHAKAL your study material.</div>', unsafe_allow_html=True)
st.markdown('<div class="ascii-note">add as many sources as you want — mix pdfs, websites, youtube & images.</div>', unsafe_allow_html=True)

col_left, col_right = st.columns(2, gap="medium")

with col_left:
    pdf_files = st.file_uploader("📄 PDF FILES", type=["pdf"], accept_multiple_files=True)
    website_text = st.text_area("🌐 WEBSITE URLS", placeholder="one url per line...")

with col_right:
    image_files = st.file_uploader("🖼️ IMAGE FILES", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True)
    youtube_text = st.text_area("🎥 YOUTUBE URLS", placeholder="one url per line...")

website_urls = [x.strip() for x in website_text.splitlines() if x.strip()]
youtube_urls = [x.strip() for x in youtube_text.splitlines() if x.strip()]

total_sources = (
    len(pdf_files or [])
    + len(website_urls)
    + len(youtube_urls)
    + len(image_files or [])
)

st.caption(f"> {total_sources} source(s) queued")

process_button = st.button("[ PROCESS_SOURCES ]", use_container_width=True, type="primary")

if process_button:
    if total_sources == 0:
        st.warning("> no sources detected. add at least one.")
    else:
        all_docs = []
        source_names = []
        progress = st.progress(0)
        status = st.empty()
        first_status = {"value": True}

        def show_status(main_text):
            if first_status["value"]:
                extra = '<div class="processing-message">aniket bhai ki taraf se hello! :)</div>'
                first_status["value"] = False
            else:
                extra = f'<div class="processing-fact">{html.escape(next_fact())}</div>'

            status.markdown(
                '<div class="processing">'
                f'<div class="processing-main">{html.escape(main_text)}</div>'
                f'{extra}'
                '</div>',
                unsafe_allow_html=True
            )

        try:
            current = 0

            if pdf_files:
                for file in pdf_files:
                    show_status(f"Processing {file.name}...")
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp:
                        temp.write(file.getbuffer())
                        temp_path = temp.name
                    try:
                        docs = load_pdf(temp_path)
                        for doc in docs:
                            doc.metadata["source_type"] = "pdf"
                            doc.metadata["source"] = file.name
                        all_docs.extend(docs)
                        source_names.append(f"📄 {file.name}")
                    finally:
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                    current += 1
                    progress.progress(current / total_sources)

            for url in website_urls:
                show_status("Scraping website...")
                docs = load_webpage(url)
                for doc in docs:
                    doc.metadata["source_type"] = "webpage"
                    doc.metadata["source"] = url
                all_docs.extend(docs)
                source_names.append(f"🌐 {url}")
                current += 1
                progress.progress(current / total_sources)

            for url in youtube_urls:
                show_status("Extracting YouTube transcript...")
                docs = load_youtube(url)
                all_docs.extend(docs)
                source_names.append("🎥 YouTube")
                current += 1
                progress.progress(current / total_sources)

            if image_files:
                for file in image_files:
                    show_status(f"Running OCR on {file.name}...")
                    extension = os.path.splitext(file.name)[1]
                    with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as temp:
                        temp.write(file.getbuffer())
                        temp_path = temp.name
                    try:
                        docs = load_image(temp_path)
                        for doc in docs:
                            doc.metadata["source_type"] = "image"
                            doc.metadata["source"] = file.name
                        all_docs.extend(docs)
                        source_names.append(f"🖼️ {file.name}")
                    finally:
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                    current += 1
                    progress.progress(current / total_sources)

            show_status("Building vector embeddings...")
            add_to_chroma(all_docs)
            progress.progress(1.0)
            status.success("> sources indexed and ready.")
            st.session_state.sources = source_names
            st.session_state.source_ready = True

        except Exception as error:
            status.empty()
            st.error(f"> ERROR: {error}")

if st.session_state.source_ready:
    st.markdown(
        '<div class="source-ready">'
        '<div class="source-ready-title">SOURCE READY</div>'
        '<div class="source-ready-text">your study material is now available to SHAKAL.</div>'
        '</div>',
        unsafe_allow_html=True
    )
    num_sources = len(st.session_state.sources)
    if num_sources > 0:
        chip_cols = st.columns(num_sources)
        for col, source in zip(chip_cols, st.session_state.sources):
            with col:
                st.markdown(f'<div class="source-chip">{html.escape(source)}</div>', unsafe_allow_html=True)

# ==========================================
# PERSONALITY
# ==========================================
st.markdown('<div class="ascii-section">PERSONALITY_MODE</div>', unsafe_allow_html=True)
st.markdown('<div class="ascii-subsection">choose how SHAKAL should teach you.</div>', unsafe_allow_html=True)

personalities = ["Normal", "Theorist", "Practicalist", "Examiner", "Guide"]

selected_personality = st.radio(
    "Personality",
    personalities,
    index=personalities.index(st.session_state.personality),
    label_visibility="collapsed",
    key="personality_selector"
)

st.session_state.personality = selected_personality

st.markdown(f'<div class="personality-info">&gt; active_mode: {st.session_state.personality}</div>', unsafe_allow_html=True)

# ==========================================
# STUDY CHAT
# ==========================================
st.markdown('<div class="ascii-section">STUDY_CHAT</div>', unsafe_allow_html=True)

reset_col, status_col = st.columns([1, 4])
with reset_col:
    if st.button("[ RESET_CHAT ]", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

with status_col:
    st.caption(f"> mode: {st.session_state.personality}")

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

query = st.chat_input("user@shakal:~$ ask anything...")

PERSONALITY_PROMPTS = {
    "Normal": "You are SHAKAL, a balanced AI study assistant. Explain concepts clearly with examples, analogies, and step-by-step reasoning. Adapt to the question.",
    "Theorist": "You are SHAKAL in Theorist mode. Focus on underlying theory, definitions, principles, relationships, mathematical reasoning, and WHY things work.",
    "Practicalist": "You are SHAKAL in Practicalist mode. Focus on real-world applications, implementation, workflows, demonstrations, use cases, and common mistakes.",
    "Examiner": "You are SHAKAL in Examiner mode. Act like a strict academic examiner. Focus on exam questions, conceptual gaps, mistake analysis, and useful feedback. No unnecessary praise.",
    "Guide": "You are SHAKAL in Guide mode. Act as a Socratic mentor. Break topics into steps, ask guiding questions, build independent problem-solving skills. Don't just give answers."
}

if query:
    st.session_state.chat_history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        processing = st.empty()

        processing.markdown(
            '<div class="processing">'
            '<div class="processing-main">INITIALIZING</div>'
            '<div class="processing-message">aniket bhai ki taraf se hello! :)</div>'
            '</div>',
            unsafe_allow_html=True
        )
        time.sleep(1.5)

        processing.markdown(
            '<div class="processing">'
            '<div class="processing-main">PROCESSING QUERY</div>'
            '<div class="processing-message">aniket bhai ki taraf se hello! :)</div>'
            f'<div class="processing-fact">{html.escape(next_fact())}</div>'
            '</div>',
            unsafe_allow_html=True
        )

        try:
            llm = ChatMistralAI(model="mistral-small-2506")
            personality_prompt = PERSONALITY_PROMPTS[st.session_state.personality]

            if not st.session_state.source_ready:
                system_prompt = f"{personality_prompt}\n\nUse general knowledge. Explain step-by-step with examples."
                prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{question}")])
                final_prompt = prompt.invoke({"question": query})
            else:
                embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embedding_model)
                retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5})
                docs = retriever.invoke(query)
                context = "\n\n".join(doc.page_content for doc in docs)

                system_prompt = f"{personality_prompt}\n\nUse the student's sources as PRIMARY. If general knowledge is needed, clearly say so."
                prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "Context:\n{context}\n\nQuestion:\n{question}")])
                final_prompt = prompt.invoke({"context": context, "question": query})

            response = llm.invoke(final_prompt)
            answer = response.content
            processing.empty()
            st.write(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})

        except Exception as error:
            processing.empty()
            st.error(f"> SYSTEM ERROR: {error}")

st.markdown(
    '<pre class="ascii-hr">════════════════════════════════════════════════════════════</pre>'
    '<div class="personality-info">&gt; EOF &nbsp;|&nbsp; connection terminated &nbsp;|&nbsp; SHAKAL v1.0</div>',
    unsafe_allow_html=True
            )
