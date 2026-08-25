import os
import random
import tempfile
import itertools

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
    page_title="SHAKAL - STUDY MADAD",
    page_icon="🧠",
    layout="wide"
)

# ==========================================
# AI QUOTES & FACTS (shown while processing)
# ==========================================
AI_QUOTES = [
    "“The best way to predict the future is to invent it.” — Alan Kay",
    "“Intelligence is the ability to adapt to change.” — Stephen Hawking",
    "“AI is probably the most important thing humanity has ever worked on.” — Sundar Pichai",
    "“Machine intelligence is the last invention humanity will ever need to make.” — Nick Bostrom",
    "⚡ Fun fact: The term “Artificial Intelligence” was coined in 1956 at the Dartmouth Conference.",
    "⚡ Fun fact: The first chatbot, ELIZA, was built at MIT in 1966.",
    "⚡ Fun fact: Neural networks are inspired by the neurons of the human brain.",
    "⚡ Fun fact: Over 90% of the world's data was created in just the last few years.",
    "⚡ Fun fact: RAG (Retrieval-Augmented Generation) is exactly how SHAKAL reads your sources!",
    "“The question of whether machines can think is like the question of whether submarines can swim.” — Edsger Dijkstra",
]

# ==========================================
# MODERN CSS INJECTION
# ==========================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    #MainMenu, footer, header, .stDeployButton {
        visibility: hidden;
    }

    .block-container {
        max-width: 1100px;
        padding-top: 4rem;
        padding-bottom: 4rem;
    }

    /* ===== RESPONSIVE BRANDING ===== */
    .brand {
        font-family: 'Space Grotesk', sans-serif;
        text-align: center;
        font-size: clamp(38px, 10vw, 80px);
        font-weight: 700;
        letter-spacing: clamp(4px, 1.6vw, 16px);
        line-height: 1.1;
        white-space: nowrap;
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 50%, #667eea 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }

    .subtitle {
        font-family: 'Space Grotesk', sans-serif;
        text-align: center;
        font-size: clamp(13px, 2.6vw, 20px);
        font-weight: 500;
        letter-spacing: clamp(6px, 2.2vw, 18px);
        margin-left: clamp(6px, 2.2vw, 18px);
        color: rgba(255, 255, 255, 0.5);
        text-transform: uppercase;
        white-space: nowrap;
    }

    .tagline {
        text-align: center;
        font-size: 16px;
        color: rgba(255, 255, 255, 0.6);
        margin-top: 30px;
        margin-bottom: 12px;
    }

    .model {
        text-align: center;
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        color: #4facfe;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-top: 8px;
        opacity: 0.8;
    }

    .developer {
        text-align: center;
        font-size: 13px;
        color: rgba(255, 255, 255, 0.4);
        margin-top: 30px;
        margin-bottom: 20px;
    }

    .dev-name {
        color: rgba(255, 255, 255, 0.8);
        font-weight: 600;
    }

    .developer a {
        text-decoration: none;
        color: #4facfe;
        transition: all 0.3s ease;
        font-weight: 500;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        vertical-align: middle;
    }

    .developer a:hover {
        color: #00f2fe;
        transform: translateY(-1px);
    }

    /* ===== SECTIONS ===== */
    .section-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(20px, 4vw, 24px);
        font-weight: 600;
        margin-top: 60px;
        margin-bottom: 12px;
        padding-left: 16px;
        border-left: 4px solid;
        border-image: linear-gradient(to bottom, #00f2fe, #4facfe) 1;
        color: #ffffff;
    }

    .section-subtitle {
        font-size: 15px;
        color: rgba(255, 255, 255, 0.5);
        margin-bottom: 28px;
        margin-left: 16px;
    }

    .source-note {
        padding: 16px 20px;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.08);
        background: rgba(255,255,255,0.02);
        backdrop-filter: blur(10px);
        font-size: 14px;
        color: rgba(255, 255, 255, 0.7);
        margin-bottom: 28px;
        margin-left: 16px;
    }

    /* ===== PERSONALITY PILLS — SINGLE CLICK, NO CUT NAMES ===== */

    /* Hide the stray main widget label box */
    [data-testid="stRadio"] > label,
    [data-testid="stRadio"] > div > label {
        display: none !important;
    }

    /* Pill container: wraps to next line on mobile instead of squeezing */
    [data-testid="stRadio"] div[role="radiogroup"] {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: wrap !important;
        gap: 10px !important;
        justify-content: center !important;
    }

    /* Individual pills */
    [data-testid="stRadio"] div[role="radiogroup"] label {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 999px !important;
        padding: 10px 22px !important;
        margin: 0 !important;
        color: rgba(255, 255, 255, 0.8) !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
        font-size: 14px !important;
        white-space: nowrap !important;   /* <- names NEVER split mid-word */
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
        backdrop-filter: blur(8px) !important;
        cursor: pointer !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
    }

    [data-testid="stRadio"] div[role="radiogroup"] label:hover {
        border-color: #00f2fe !important;
        background: rgba(0, 242, 254, 0.08) !important;
        color: #ffffff !important;
        box-shadow: 0 0 15px rgba(0, 242, 254, 0.15) !important;
    }

    /* Nuke the native radio circle completely */
    [data-testid="stRadio"] div[role="radiogroup"] svg,
    [data-testid="stRadio"] div[role="radiogroup"] input[type="radio"],
    [data-testid="stRadio"] div[role="radiogroup"] [data-baseweb="radio"] {
        display: none !important;
        width: 0 !important;
        height: 0 !important;
        margin: 0 !important;
    }

    [data-testid="stRadio"] div[role="radiogroup"] label > div {
        padding: 0 !important;
        margin: 0 !important;
    }

    /* Selected pill (instant single-click visual update) */
    [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%) !important;
        color: #0b0f19 !important;
        border-color: transparent !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 15px rgba(0, 242, 254, 0.25) !important;
    }

    /* ===== PROCESSING BOX ===== */
    .processing {
        text-align: center;
        padding: 24px;
        border-radius: 16px;
        border: 1px solid rgba(79, 172, 254, 0.2);
        background: rgba(79, 172, 254, 0.05);
        margin: 20px 0;
        position: relative;
        overflow: hidden;
    }

    .processing::after {
        content: "";
        position: absolute;
        top: -50%; left: -50%;
        width: 200%; height: 200%;
        background: linear-gradient(to bottom right,
            rgba(255,255,255,0) 0%,
            rgba(255,255,255,0.05) 50%,
            rgba(255,255,255,0) 100%);
        transform: rotate(45deg);
        animation: shimmer 2.5s infinite;
    }

    @keyframes shimmer {
        0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
        100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
    }

    .processing-main {
        font-weight: 600;
        font-size: 18px;
        color: #4facfe;
        position: relative;
        z-index: 1;
    }

    .processing-message {
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        color: rgba(255, 255, 255, 0.4);
        margin-top: 8px;
        position: relative;
        z-index: 1;
    }

    .processing-fact {
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        font-style: italic;
        color: #00f2fe;
        margin-top: 12px;
        padding-top: 12px;
        border-top: 1px dashed rgba(255,255,255,0.15);
        position: relative;
        z-index: 1;
    }

    /* ===== SOURCE READY & CHIPS ===== */
    .source-ready {
        padding: 20px 24px;
        border-radius: 12px;
        border: 1px solid rgba(0, 242, 254, 0.2);
        background: rgba(0, 242, 254, 0.03);
        margin-top: 24px;
        margin-bottom: 16px;
        position: relative;
        overflow: hidden;
    }

    .source-ready::before {
        content: '';
        position: absolute;
        top: 0; left: 0;
        width: 100%; height: 2px;
        background: linear-gradient(90deg, transparent, #00f2fe, transparent);
    }

    .source-ready-title {
        font-weight: 700;
        font-size: 14px;
        color: #00f2fe;
        letter-spacing: 1.5px;
        display: flex;
        align-items: center;
    }

    .source-ready-text {
        font-size: 13px;
        color: rgba(255, 255, 255, 0.6);
        margin-top: 6px;
    }

    .source-chip {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 8px 16px;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        font-size: 13px;
        font-weight: 500;
        color: rgba(255, 255, 255, 0.8);
        text-align: center;
        width: 100%;
    }

    .personality-info {
        text-align: center;
        font-size: 13px;
        color: rgba(255, 255, 255, 0.3);
        margin-top: 20px;
        font-family: 'JetBrains Mono', monospace;
    }

    /* ===== BUTTONS ===== */
    .stButton > button {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.15);
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(8px);
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        color: rgba(255, 255, 255, 0.8);
        padding: 10px 20px;
    }

    .stButton > button:hover {
        border-color: #00f2fe;
        background: rgba(0, 242, 254, 0.08);
        color: #ffffff;
        box-shadow: 0 0 15px rgba(0, 242, 254, 0.15);
    }

    .stButton > button[kind="primary"],
    .stButton > button[data-testid="stBaseButton-primary"],
    .stButton > button[data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%) !important;
        color: #0b0f19 !important;
        border: none !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 15px rgba(0, 242, 254, 0.2);
    }

    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="stBaseButton-primary"]:hover,
    .stButton > button[data-testid="baseButton-primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 242, 254, 0.4) !important;
        filter: brightness(1.1);
    }

    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #00f2fe, #4facfe);
    }

    /* ===== CHAT INPUT (centered placeholder) ===== */
    .stChatInput textarea,
    [data-testid="stChatInputTextArea"] {
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        background: rgba(255, 255, 255, 0.03) !important;
        min-height: 52px !important;
        line-height: 24px !important;
        padding: 14px 16px !important;
    }

    .stChatInput textarea::placeholder,
    [data-testid="stChatInputTextArea"]::placeholder {
        line-height: 24px !important;
        vertical-align: middle !important;
    }

    .stChatInput textarea:focus,
    [data-testid="stChatInputTextArea"]:focus {
        border-color: #00f2fe !important;
        box-shadow: 0 0 0 2px rgba(0, 242, 254, 0.2) !important;
    }

    .stChatMessage, [data-testid="stChatMessage"] {
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 12px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ==========================================
# STATE INITIALIZATION
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
# HEADER & BRANDING
# ==========================================
st.markdown(
    """
    <div class="brand">SHAKAL</div>
    <div class="subtitle">STUDY MADAD</div>
    <div class="tagline">Your AI Study Assistant</div>
    <div class="model">Model · Mistral Small 2506</div>
    <div class="developer">
        Developed by <span class="dev-name">Aniket Sharma</span>
        <br><br>
        <a href="https://www.linkedin.com/in/aniket-sharma-42a700418?utm_source=share_via&utm_content=member_android" target="_blank">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M16 8a6 6 0 0 1 6 6v7h-4v-7a2 2 0 0 0-2-2 2 2 0 0 0-2 2v7h-4v-7a6 6 0 0 1 6-6z"></path><rect x="2" y="9" width="4" height="12"></rect><circle cx="4" cy="4" r="2"></circle></svg>
            LinkedIn
        </a>
        &nbsp;&nbsp;·&nbsp;&nbsp;
        <a href="https://github.com/aniket-andyy" target="_blank">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22"></path></svg>
            GitHub
        </a>
    </div>
    """,
    unsafe_allow_html=True
)

# ==========================================
# UPLOAD SOURCES (ALL TYPES AT ONCE)
# ==========================================
st.markdown('<div class="section-title">Upload Your Sources</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subtitle">Give SHAKAL your study material.</div>', unsafe_allow_html=True)
st.markdown('<div class="source-note">⚠️ You can process up to <b>3 sources</b> at a time — mix & match any type below.</div>', unsafe_allow_html=True)

col_left, col_right = st.columns(2, gap="medium")

with col_left:
    pdf_files = st.file_uploader("📄 PDF Files", type=["pdf"], accept_multiple_files=True)
    website_text = st.text_area("🌐 Website URLs", placeholder="Paste one URL per line...")

with col_right:
    image_files = st.file_uploader("🖼️ Image Files", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True)
    youtube_text = st.text_area("🎥 YouTube URLs", placeholder="Paste one YouTube URL per line...")

website_urls = [x.strip() for x in website_text.splitlines() if x.strip()]
youtube_urls = [x.strip() for x in youtube_text.splitlines() if x.strip()]

total_sources = (
    len(pdf_files or [])
    + len(website_urls)
    + len(youtube_urls)
    + len(image_files or [])
)

st.caption(f"{total_sources}/3 sources selected")

process_button = st.button("⚡ Process Sources", use_container_width=True, type="primary")

if process_button:
    if total_sources == 0:
        st.warning("Please add at least one source.")
    elif total_sources > 3:
        st.error("Maximum 3 sources can be processed at a time.")
    else:
        all_docs = []
        source_names = []
        progress = st.progress(0)
        status = st.empty()
        fact_cycle = itertools.cycle(AI_QUOTES)

        def show_status(main_text):
            status.markdown(
                f"""
                <div class="processing">
                    <div class="processing-main">{main_text}</div>
                    <div class="processing-fact">💡 {next(fact_cycle)}</div>
                </div>
                """,
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
                show_status("Processing website...")
                docs = load_webpage(url)
                for doc in docs:
                    doc.metadata["source_type"] = "webpage"
                    doc.metadata["source"] = url
                all_docs.extend(docs)
                source_names.append(f"🌐 {url}")
                current += 1
                progress.progress(current / total_sources)

            for url in youtube_urls:
                show_status("Processing YouTube transcript...")
                docs = load_youtube(url)
                all_docs.extend(docs)
                source_names.append("🎥 YouTube")
                current += 1
                progress.progress(current / total_sources)

            if image_files:
                for file in image_files:
                    show_status(f"Processing {file.name}...")
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

            show_status("Creating embeddings and updating knowledge base...")
            add_to_chroma(all_docs)
            progress.progress(1.0)
            status.success("Sources ready.")
            st.session_state.sources = source_names
            st.session_state.source_ready = True

        except Exception:
            status.empty()
            st.error("Unable to process the selected source.")

if st.session_state.source_ready:
    st.markdown(
        """
        <div class="source-ready">
            <div class="source-ready-title">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 8px;"><polyline points="20 6 9 17 4 12"></polyline></svg>
                SOURCE READY
            </div>
            <div class="source-ready-text">Your study material is available to SHAKAL.</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    num_sources = len(st.session_state.sources)
    if num_sources > 0:
        chip_cols = st.columns(num_sources)
        for col, source in zip(chip_cols, st.session_state.sources):
            with col:
                st.markdown(f'<div class="source-chip">{source}</div>', unsafe_allow_html=True)

# ==========================================
# PERSONALITY SELECTION (SINGLE-CLICK, NO CUT NAMES)
# ==========================================
st.markdown('<div class="section-title">Personality</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subtitle">Choose how SHAKAL should teach you.</div>', unsafe_allow_html=True)

personalities = ["Normal", "Theorist", "Practicalist", "Examiner", "Guide"]

selected_personality = st.radio(
    "Personality",
    personalities,
    index=personalities.index(st.session_state.personality),
    label_visibility="collapsed",
    key="personality_selector"
)

st.session_state.personality = selected_personality

st.markdown(f'<div class="personality-info">Active personality · {st.session_state.personality}</div>', unsafe_allow_html=True)

# ==========================================
# STUDY CHAT
# ==========================================
st.markdown('<div class="section-title">Study Chat</div>', unsafe_allow_html=True)

reset_col, status_col = st.columns([1, 4])
with reset_col:
    if st.button("Reset Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

with status_col:
    st.caption(f"Personality: {st.session_state.personality}")

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

query = st.chat_input("Ask SHAKAL anything about your studies...")

PERSONALITY_PROMPTS = {
    "Normal": """You are SHAKAL, a balanced AI study assistant. Help students and learners understand subjects clearly. Explain concepts accurately and at an appropriate level of detail. Use examples, analogies, step-by-step explanations, and concise summaries when useful. Adapt your teaching style to the user's question. Do not unnecessarily overcomplicate simple questions.""",
    "Theorist": """You are SHAKAL in Theorist personality. Focus on underlying theory and conceptual foundations. Explain definitions, principles, relationships between concepts, assumptions, mathematical reasoning, cause and effect, and why something works. Prefer deep conceptual understanding over quick answers.""",
    "Practicalist": """You are SHAKAL in Practicalist personality. Focus on how knowledge is used in the real world. Explain practical applications, implementation, real-world examples, workflows, demonstrations, use cases, and common mistakes. Whenever useful, convert theory into something the learner can actually do or implement.""",
    "Examiner": """You are SHAKAL in Examiner personality. Act like a strict but helpful academic examiner. Focus on important concepts, likely exam questions, conceptual gaps, mistakes, problem-solving, and evaluation. When the user provides an answer, evaluate it, identify mistakes, explain why they are mistakes, and show how to improve. Do not praise unnecessarily. Focus on useful feedback.""",
    "Guide": """You are SHAKAL in Guide personality. Act as a study mentor who guides the learner through a topic. Help the student decide what to learn first, what to learn next, what to practice, and how to improve. Break difficult topics into manageable steps. Ask guiding questions when useful. Identify knowledge gaps and suggest the next logical learning step. Your goal is to help the student become capable of solving problems independently rather than simply giving answers."""
}

if query:
    st.session_state.chat_history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        processing = st.empty()
        processing.markdown(
            f"""
            <div class="processing">
                <div class="processing-main">Processing</div>
                <div class="processing-message">[ aniket bhai ki taraf se hello! :) ]</div>
                <div class="processing-fact">💡 {random.choice(AI_QUOTES)}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        try:
            llm = ChatMistralAI(model="mistral-small-2506")
            personality_prompt = PERSONALITY_PROMPTS[st.session_state.personality]

            if not st.session_state.source_ready:
                system_prompt = f"""{personality_prompt}\n\nYou are helping a student using your general knowledge. Explain difficult concepts clearly and step-by-step. Use examples, analogies, and practical explanations when useful."""
                prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{question}")])
                final_prompt = prompt.invoke({"question": query})
            else:
                embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embedding_model)
                retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5})
                docs = retriever.invoke(query)
                context = "\n\n".join(doc.page_content for doc in docs)

                system_prompt = f"""{personality_prompt}\n\nYou are answering using the student's provided study sources. Use the provided source as the PRIMARY source. You may use your general knowledge when the source does not contain enough information. If you use general knowledge, clearly state that the information is not directly present in the provided source. Help the student understand the topic accurately and clearly."""
                prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "Study Source Context:\n\n{context}\n\nStudent Question:\n\n{question}")])
                final_prompt = prompt.invoke({"context": context, "question": query})

            response = llm.invoke(final_prompt)
            answer = response.content
            processing.empty()
            st.write(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})

        except Exception as error:
            status.empty()
            st.error(f"Unable to process the selected source.\n\n🔍 Debug: {error}")
