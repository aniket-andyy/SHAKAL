import os
import tempfile

import streamlit as st

from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

from database import (
    load_pdf,
    load_webpage,
    load_youtube,
    load_image,
    add_to_chroma
)


load_dotenv()

if "MISTRAL_API_KEY" in st.secrets:
    os.environ["MISTRAL_API_KEY"] = st.secrets["MISTRAL_API_KEY"]


st.set_page_config(
    page_title="SHAKAL - STUDY MADAD",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed"
)


st.markdown(
    """
    <style>

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }

    .stApp {
        background: #08090c;
        color: #f5f5f5;
    }

    .block-container {
        max-width: 1050px;
        padding-top: 2.5rem;
        padding-bottom: 4rem;
    }

    .hero {
        text-align: center;
        padding: 30px 0 20px 0;
    }

    .brand {
        font-size: 52px;
        font-weight: 800;
        letter-spacing: 8px;
        line-height: 1;
        color: #ffffff;
    }

    .subtitle {
        font-size: 15px;
        letter-spacing: 5px;
        margin-top: 12px;
        color: #8f96a3;
        font-weight: 600;
    }

    .tagline {
        margin-top: 20px;
        font-size: 18px;
        color: #c7cbd3;
    }

    .model {
        display: inline-block;
        margin-top: 12px;
        padding: 7px 14px;
        border: 1px solid #292d35;
        border-radius: 20px;
        background: #111318;
        color: #9da4b0;
        font-size: 13px;
    }

    .developer {
        margin-top: 18px;
        color: #777f8d;
        font-size: 13px;
    }

    .developer a {
        color: #b9c0cb;
        text-decoration: none;
        margin: 0 7px;
    }

    .developer a:hover {
        color: #ffffff;
    }

    .section-title {
        font-size: 24px;
        font-weight: 700;
        color: #ffffff;
        margin-top: 38px;
        margin-bottom: 5px;
    }

    .section-description {
        color: #777f8d;
        margin-bottom: 20px;
        font-size: 14px;
    }

    .source-card {
        background: #101217;
        border: 1px solid #252a33;
        border-radius: 16px;
        padding: 22px;
        margin-top: 10px;
    }

    .source-ready {
        background: #101712;
        border: 1px solid #263d2b;
        border-radius: 12px;
        padding: 14px 18px;
        margin-top: 18px;
        color: #b7d8bd;
    }

    .processing {
        background: #111318;
        border: 1px solid #292d35;
        border-radius: 14px;
        padding: 18px;
        margin: 15px 0;
        text-align: center;
    }

    .processing-title {
        font-size: 16px;
        font-weight: 700;
        color: #ffffff;
    }

    .processing-message {
        color: #777f8d;
        font-size: 13px;
        margin-top: 5px;
    }

    .mode-description {
        text-align: center;
        color: #777f8d;
        font-size: 13px;
        margin-top: 8px;
        margin-bottom: 20px;
    }

    .chat-box {
        background: #101217;
        border: 1px solid #252a33;
        border-radius: 16px;
        padding: 18px;
        margin: 12px 0;
    }

    .user-label {
        color: #8f96a3;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1px;
        margin-bottom: 7px;
    }

    .ai-label {
        color: #ffffff;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1px;
        margin-bottom: 7px;
    }

    .ai-processing {
        color: #9da4b0;
        font-size: 13px;
        padding: 12px 0;
    }

    .source-meta {
        color: #7f8794;
        font-size: 12px;
        margin-top: 4px;
    }

    div[data-testid="stFileUploader"] {
        background: #0d0f13;
        border-radius: 12px;
    }

    div[data-testid="stTextInput"] input,
    div[data-testid="stTextArea"] textarea {
        background: #0d0f13;
        color: #ffffff;
        border: 1px solid #292e37;
        border-radius: 10px;
    }

    button[kind="primary"] {
        border-radius: 10px;
        font-weight: 700;
    }

    div[role="radiogroup"] {
        gap: 8px;
    }

    div[role="radiogroup"] label {
        background: #101217;
        border: 1px solid #292e37;
        border-radius: 10px;
        padding: 12px 18px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


if "mode" not in st.session_state:
    st.session_state.mode = "Source + LLM/AI"

if "source_ready" not in st.session_state:
    st.session_state.source_ready = False

if "source_name" not in st.session_state:
    st.session_state.source_name = None

if "source_type" not in st.session_state:
    st.session_state.source_type = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def render_header():

    st.markdown(
        """
        <div class="hero">

            <div class="brand">
                SHAKAL
            </div>

            <div class="subtitle">
                STUDY MADAD
            </div>

            <div class="tagline">
                Your AI Study Assistant
            </div>

            <div class="model">
                Model · Mistral Small 2506
            </div>

            <div class="developer">
                Developed by Aniket Sharma
                <br>
                <a href="https://www.linkedin.com/in/aniket-sharma-42a700418?utm_source=share_via&utm_content=member_android" target="_blank">
                    LinkedIn
                </a>
                |
                <a href="https://github.com/aniket-andyy" target="_blank">
                    GitHub
                </a>
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


def render_source_section():

    st.markdown(
        '<div class="section-title">Upload Your Source</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">Give SHAKAL your study material and start learning.</div>',
        unsafe_allow_html=True
    )

    source_type = st.radio(
        "Source type",
        [
            "📄 PDF",
            "🌐 Website Link",
            "🎥 YouTube Video",
            "🖼️ Image"
        ],
        horizontal=True,
        label_visibility="collapsed"
    )

    st.markdown('<div class="source-card">', unsafe_allow_html=True)

    uploaded_file = None
    url = None

    if source_type == "📄 PDF":

        uploaded_file = st.file_uploader(
            "Upload PDF",
            type=["pdf"],
            label_visibility="visible"
        )

        if uploaded_file:
            st.caption(
                f"{uploaded_file.name} · "
                f"{uploaded_file.size / 1024:.1f} KB"
            )

    elif source_type == "🌐 Website Link":

        url = st.text_input(
            "Paste Website URL",
            placeholder="https://example.com/article"
        )

    elif source_type == "🎥 YouTube Video":

        url = st.text_input(
            "Paste YouTube Video URL",
            placeholder="https://youtube.com/watch?v=..."
        )

    elif source_type == "🖼️ Image":

        uploaded_file = st.file_uploader(
            "Upload Image",
            type=["png", "jpg", "jpeg", "webp"],
            label_visibility="visible"
        )

        if uploaded_file:
            st.caption(
                f"{uploaded_file.name} · "
                f"{uploaded_file.size / 1024:.1f} KB"
            )

    process = st.button(
        "Process Source",
        type="primary",
        use_container_width=True
    )

    st.markdown("</div>", unsafe_allow_html=True)

    if process:

        try:

            with st.status(
                "Processing source...",
                expanded=True
            ) as status:

                st.write("Extracting content...")

                if source_type == "📄 PDF":

                    if not uploaded_file:
                        st.error("Please upload a PDF.")
                        return

                    suffix = ".pdf"

                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=suffix
                    ) as temp:

                        temp.write(
                            uploaded_file.getbuffer()
                        )

                        temp_path = temp.name

                    docs = load_pdf(temp_path)

                    st.write("Creating embeddings...")

                    add_to_chroma(docs)

                    os.unlink(temp_path)

                    st.session_state.source_name = uploaded_file.name
                    st.session_state.source_type = "📄 PDF"

                elif source_type == "🌐 Website Link":

                    if not url:
                        st.error("Please enter a website URL.")
                        return

                    docs = load_webpage(url)

                    st.write("Creating embeddings...")

                    add_to_chroma(docs)

                    st.session_state.source_name = url
                    st.session_state.source_type = "🌐 Website"

                elif source_type == "🎥 YouTube Video":

                    if not url:
                        st.error("Please enter a YouTube URL.")
                        return

                    docs = load_youtube(url)

                    st.write("Creating embeddings...")

                    add_to_chroma(docs)

                    st.session_state.source_name = url
                    st.session_state.source_type = "🎥 YouTube"

                elif source_type == "🖼️ Image":

                    if not uploaded_file:
                        st.error("Please upload an image.")
                        return

                    suffix = os.path.splitext(
                        uploaded_file.name
                    )[1]

                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=suffix
                    ) as temp:

                        temp.write(
                            uploaded_file.getbuffer()
                        )

                        temp_path = temp.name

                    st.write("Extracting text with Mistral OCR...")

                    docs = load_image(temp_path)

                    st.write("Creating embeddings...")

                    add_to_chroma(docs)

                    os.unlink(temp_path)

                    st.session_state.source_name = uploaded_file.name
                    st.session_state.source_type = "🖼️ Image"

                st.session_state.source_ready = True

                status.update(
                    label="Source ready.",
                    state="complete"
                )

            st.rerun()

        except Exception:

            st.error(
                "Unable to process this source. "
                "Please check the file or URL and try again."
            )


def render_source_status():

    if st.session_state.source_ready:

        st.markdown(
            f"""
            <div class="source-ready">
                <strong>SOURCE READY</strong>
                <div class="source-meta">
                    {st.session_state.source_type}
                    &nbsp;·&nbsp;
                    {st.session_state.source_name}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


def render_mode_selector():

    st.markdown(
        '<div class="section-title">AI Working Mode</div>',
        unsafe_allow_html=True
    )

    modes = [
        "Source + LLM/AI",
        "Only My Source",
        "Only LLM/AI"
    ]

    selected = st.radio(
        "Working mode",
        modes,
        index=modes.index(
            st.session_state.mode
        ),
        horizontal=True,
        label_visibility="collapsed"
    )

    st.session_state.mode = selected

    descriptions = {
        "Source + LLM/AI":
            "Uses your source as the primary knowledge and AI knowledge when needed.",

        "Only My Source":
            "Answers strictly from your uploaded source.",

        "Only LLM/AI":
            "Answers using the AI's general knowledge without your source."
    }

    st.markdown(
        f"""
        <div class="mode-description">
            {descriptions[selected]}
        </div>
        """,
        unsafe_allow_html=True
    )


def get_rag_components():

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=embedding_model
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )

    llm = ChatMistralAI(
        model="mistral-small-2506"
    )

    return retriever, llm


def generate_response(query):

    retriever, llm = get_rag_components()

    if st.session_state.mode == "Only LLM/AI":

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a Study AI Assistant designed to help students
learn, understand, revise, and explore academic topics.

Your goal is to make studying easier and more effective.

Answer the student's question using your general knowledge.

Explain difficult concepts clearly and step-by-step.

When useful, provide examples, analogies, key points,
practical applications, comparisons, and short summaries.

There is no external study source available."""
                ),
                (
                    "human",
                    "{question}"
                )
            ]
        )

        final_prompt = prompt.invoke(
            {"question": query}
        )

        return llm.invoke(
            final_prompt
        ).content

    docs = retriever.invoke(query)

    if not docs:

        return (
            "I could not find the answer "
            "in the provided source."
        )

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    if st.session_state.mode == "Only My Source":

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a Study AI Assistant designed to help students
understand and learn from their provided study material.

You MUST answer using ONLY the provided source.

Do NOT use your general knowledge.

Do NOT invent, assume, or add information that is not supported
by the provided source.

Explain the answer clearly and in a student-friendly way.

If the answer is not present in the source, say exactly:

"I could not find the answer in the provided source." """
                ),
                (
                    "human",
                    """Study Source Context:

{context}

Student's Question:

{question}"""
                )
            ]
        )

    else:

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a Study AI Assistant designed to help students
learn, understand, revise, and explore academic topics.

The provided context comes from the user's study source.

Use the source as the PRIMARY source for answering.

You may use general knowledge when the source does not contain
enough information.

If you use general knowledge, clearly state that it is not directly
present in the provided source.

Explain difficult concepts clearly and step-by-step."""
                ),
                (
                    "human",
                    """Study Source Context:

{context}

Student's Question:

{question}"""
                )
            ]
        )

    final_prompt = prompt.invoke(
        {
            "context": context,
            "question": query
        }
    )

    return llm.invoke(
        final_prompt
    ).content


def render_chat():

    st.markdown(
        '<div class="section-title">Study With SHAKAL</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">Ask questions, understand concepts, and learn smarter.</div>',
        unsafe_allow_html=True
    )

    for message in st.session_state.chat_history:

        if message["role"] == "user":

            st.markdown(
                f"""
                <div class="chat-box">
                    <div class="user-label">YOU</div>
                    <div>{message["content"]}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
                <div class="chat-box">
                    <div class="ai-label">SHAKAL</div>
                    <div>{message["content"]}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    query = st.text_area(
        "Prompt",
        placeholder="Ask SHAKAL anything about your studies...",
        height=120,
        label_visibility="collapsed"
    )

    send = st.button(
        "Send",
        type="primary",
        use_container_width=True
    )

    if send and query.strip():

        query = query.strip()

        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": query
            }
        )

        processing_placeholder = st.empty()

        processing_placeholder.markdown(
            """
            <div class="processing">
                <div class="processing-title">
                    Processing
                </div>
                <div class="processing-message">
                    [ aniket bhai ki taraf se hello! :) ]
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        try:

            answer = generate_response(
                query
            )

            processing_placeholder.empty()

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )

            st.rerun()

        except Exception:

            processing_placeholder.empty()

            st.error(
                "Unable to generate a response. "
                "Please check your API key and backend configuration."
            )


def main():

    render_header()

    render_source_section()

    render_source_status()

    render_mode_selector()

    render_chat()


if __name__ == "__main__":
    main()
