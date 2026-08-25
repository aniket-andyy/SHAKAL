import os
import tempfile

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

    .block-container {
        max-width: 1100px;
        padding-top: 3rem;
        padding-bottom: 4rem;
    }

    .brand {
        text-align: center;
        font-size: 58px;
        font-weight: 900;
        letter-spacing: 7px;
        margin-bottom: -5px;
    }

    .subtitle {
        text-align: center;
        font-size: 22px;
        letter-spacing: 8px;
        font-weight: 600;
        opacity: 0.75;
    }

    .tagline {
        text-align: center;
        font-size: 16px;
        margin-top: 15px;
        opacity: 0.65;
    }

    .model {
        text-align: center;
        font-size: 14px;
        margin-top: 8px;
        opacity: 0.55;
    }

    .developer {
        text-align: center;
        font-size: 13px;
        margin-top: 12px;
        opacity: 0.6;
    }

    .developer a {
        text-decoration: none;
    }

    .section-title {
        font-size: 26px;
        font-weight: 700;
        margin-top: 45px;
        margin-bottom: 5px;
    }

    .section-subtitle {
        opacity: 0.6;
        margin-bottom: 25px;
    }

    .source-card {
        padding: 22px;
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 14px;
        background: rgba(255,255,255,0.025);
        margin-bottom: 15px;
    }

    .source-title {
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .source-description {
        font-size: 13px;
        opacity: 0.55;
        margin-bottom: 15px;
    }

    .mode-card {
        padding: 20px;
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 15px;
        background: rgba(255,255,255,0.025);
        margin-top: 25px;
    }

    .processing {
        text-align: center;
        padding: 25px;
        border-radius: 15px;
        border: 1px solid rgba(255,255,255,0.10);
        margin: 20px 0;
    }

    .processing-main {
        font-size: 18px;
        font-weight: 700;
    }

    .processing-message {
        font-size: 13px;
        opacity: 0.55;
        margin-top: 5px;
    }

    .source-ready {
        padding: 15px 20px;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.12);
        margin-top: 15px;
    }

    .source-ready-title {
        font-weight: 700;
        font-size: 14px;
    }

    .source-ready-text {
        font-size: 13px;
        opacity: 0.65;
        margin-top: 4px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


if "mode" not in st.session_state:
    st.session_state.mode = "Source + LLM/AI"

if "source_ready" not in st.session_state:
    st.session_state.source_ready = False

if "sources" not in st.session_state:
    st.session_state.sources = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "processed_source_count" not in st.session_state:
    st.session_state.processed_source_count = 0


st.markdown(
    """
    <div class="brand">SHAKAL</div>

    <div class="subtitle">STUDY MADAD</div>

    <div class="tagline">
        Your AI Study Assistant
    </div>

    <div class="model">
        Model · Mistral Small 2506
    </div>

    <div class="developer">
        Developed by Aniket Sharma
        <br><br>
        <a href="https://www.linkedin.com/in/aniket-sharma-42a700418?utm_source=share_via&utm_content=member_android"
           target="_blank">
            LinkedIn
        </a>
        &nbsp;&nbsp;|&nbsp;&nbsp;
        <a href="https://github.com/aniket-andyy"
           target="_blank">
            GitHub
        </a>
    </div>
    """,
    unsafe_allow_html=True
)


st.markdown(
    '<div class="section-title">Upload Your Sources</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">'
    'Give SHAKAL your study material and start learning.'
    '</div>',
    unsafe_allow_html=True
)


pdf_files = st.file_uploader(
    "📄 PDF",
    type=["pdf"],
    accept_multiple_files=True,
    help="Upload one or more PDF files."
)


website_urls = st.text_area(
    "🌐 Website Links",
    placeholder="Paste website URLs, one per line...",
    help="You can add multiple website URLs."
)


youtube_urls = st.text_area(
    "🎥 YouTube Video Links",
    placeholder="Paste YouTube URLs, one per line...",
    help="You can add multiple YouTube URLs."
)


image_files = st.file_uploader(
    "🖼️ Images",
    type=["png", "jpg", "jpeg", "webp"],
    accept_multiple_files=True,
    help="Upload one or more images."
)


website_list = [
    url.strip()
    for url in website_urls.splitlines()
    if url.strip()
]


youtube_list = [
    url.strip()
    for url in youtube_urls.splitlines()
    if url.strip()
]


total_sources = (
    len(pdf_files or [])
    + len(website_list)
    + len(youtube_list)
    + len(image_files or [])
)


st.caption(
    f"{total_sources} source(s) selected"
)


process_button = st.button(
    "⚡ Process All Sources",
    use_container_width=True,
    type="primary"
)


if process_button:

    if total_sources == 0:

        st.warning(
            "Please add at least one source."
        )

    else:

        all_docs = []
        source_names = []

        progress = st.progress(
            0
        )

        status = st.empty()

        try:

            current = 0

            total = total_sources

            if pdf_files:

                for pdf_file in pdf_files:

                    status.info(
                        f"Processing PDF: {pdf_file.name}"
                    )

                    suffix = ".pdf"

                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=suffix
                    ) as temp:

                        temp.write(
                            pdf_file.getbuffer()
                        )

                        temp_path = temp.name

                    try:

                        docs = load_pdf(
                            temp_path
                        )

                        for doc in docs:

                            doc.metadata[
                                "source_type"
                            ] = "pdf"

                            doc.metadata[
                                "source"
                            ] = pdf_file.name

                        all_docs.extend(
                            docs
                        )

                        source_names.append(
                            f"📄 {pdf_file.name}"
                        )

                    finally:

                        if os.path.exists(
                            temp_path
                        ):
                            os.remove(
                                temp_path
                            )

                    current += 1

                    progress.progress(
                        current / total
                    )


            for url in website_list:

                status.info(
                    f"Processing website: {url}"
                )

                docs = load_webpage(
                    url
                )

                for doc in docs:

                    doc.metadata[
                        "source_type"
                    ] = "webpage"

                    doc.metadata[
                        "source"
                    ] = url

                all_docs.extend(
                    docs
                )

                source_names.append(
                    f"🌐 {url}"
                )

                current += 1

                progress.progress(
                    current / total
                )


            for url in youtube_list:

                status.info(
                    f"Processing YouTube: {url}"
                )

                docs = load_youtube(
                    url
                )

                all_docs.extend(
                    docs
                )

                source_names.append(
                    f"🎥 YouTube"
                )

                current += 1

                progress.progress(
                    current / total
                )


            if image_files:

                for image_file in image_files:

                    status.info(
                        f"Processing image: {image_file.name}"
                    )

                    extension = os.path.splitext(
                        image_file.name
                    )[1]

                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=extension
                    ) as temp:

                        temp.write(
                            image_file.getbuffer()
                        )

                        temp_path = temp.name

                    try:

                        docs = load_image(
                            temp_path
                        )

                        for doc in docs:

                            doc.metadata[
                                "source_type"
                            ] = "image"

                            doc.metadata[
                                "source"
                            ] = image_file.name

                        all_docs.extend(
                            docs
                        )

                        source_names.append(
                            f"🖼️ {image_file.name}"
                        )

                    finally:

                        if os.path.exists(
                            temp_path
                        ):
                            os.remove(
                                temp_path
                            )

                    current += 1

                    progress.progress(
                        current / total
                    )


            status.info(
                "Creating embeddings and updating knowledge base..."
            )

            add_to_chroma(
                all_docs
            )

            progress.progress(
                1.0
            )

            status.success(
                "Source processing completed."
            )

            st.session_state.source_ready = True

            st.session_state.sources = source_names

            st.session_state.processed_source_count = len(
                source_names
            )

        except Exception as e:

            status.empty()

            st.error(
                "Unable to process one or more sources."
            )

            st.caption(
                str(e)
            )


if st.session_state.source_ready:

    st.markdown(
        """
        <div class="source-ready">
            <div class="source-ready-title">
                ✓ SOURCE READY
            </div>
            <div class="source-ready-text">
                Your sources are available to SHAKAL.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    for source in st.session_state.sources:

        st.caption(
            source
        )


st.markdown(
    '<div class="section-title">AI Working Mode</div>',
    unsafe_allow_html=True
)


mode = st.radio(
    "Choose how SHAKAL should answer",
    [
        "Source + LLM/AI",
        "Only My Source",
        "Only LLM/AI"
    ],
    index=0,
    horizontal=True
)


st.session_state.mode = mode


if mode == "Source + LLM/AI":

    st.caption(
        "Uses your sources as the primary knowledge and AI knowledge when needed."
    )

elif mode == "Only My Source":

    st.caption(
        "Answers strictly from your uploaded sources."
    )

else:

    st.caption(
        "Answers using the AI's general knowledge without your sources."
    )


st.markdown(
    '<div class="section-title">Study Chat</div>',
    unsafe_allow_html=True
)


for message in st.session_state.chat_history:

    if message["role"] == "user":

        with st.chat_message(
            "user"
        ):

            st.write(
                message["content"]
            )

    else:

        with st.chat_message(
            "assistant"
        ):

            st.write(
                message["content"]
            )


query = st.chat_input(
    "Ask SHAKAL anything about your studies..."
)


if query:

    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": query
        }
    )

    with st.chat_message(
        "user"
    ):

        st.write(
            query
        )

    with st.chat_message(
        "assistant"
    ):

        processing = st.empty()

        processing.markdown(
            """
            <div class="processing">
                <div class="processing-main">
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

            embedding_model = HuggingFaceEmbeddings(
                model_name=(
                    "sentence-transformers/"
                    "all-MiniLM-L6-v2"
                )
            )

            llm = ChatMistralAI(
                model="mistral-small-2506"
            )

            if mode == "Only LLM/AI":

                prompt = ChatPromptTemplate.from_messages(
                    [
                        (
                            "system",
                            """You are SHAKAL, a helpful AI
study assistant.

Your job is to help students understand,
learn, revise and practice concepts.

Explain concepts clearly and accurately.
Use examples when useful.

Answer using your general knowledge."""
                        ),
                        (
                            "human",
                            "{question}"
                        )
                    ]
                )

                final_prompt = prompt.invoke(
                    {
                        "question": query
                    }
                )

                response = llm.invoke(
                    final_prompt
                )

                answer = response.content

            else:

                if not st.session_state.source_ready:

                    answer = (
                        "Please upload and process "
                        "a source first."
                    )

                else:

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

                    docs = retriever.invoke(
                        query
                    )

                    context = "\n\n".join(
                        doc.page_content
                        for doc in docs
                    )

                    if mode == "Only My Source":

                        prompt = ChatPromptTemplate.from_messages(
                            [
                                (
                                    "system",
                                    """You are SHAKAL,
a study assistant.

You MUST answer using ONLY the
provided source context.

Do NOT use general knowledge.

Do NOT invent information.

If the answer is not present in
the source, say exactly:

"I could not find the answer in the provided source."
"""
                                ),
                                (
                                    "human",
                                    """Source Context:

{context}

Question:

{question}"""
                                )
                            ]
                        )

                    else:

                        prompt = ChatPromptTemplate.from_messages(
                            [
                                (
                                    "system",
                                    """You are SHAKAL,
a helpful AI study assistant.

Use the user's source as the
PRIMARY source for answering.

You may use your general knowledge
when the source does not contain
enough information.

If you use information not directly
present in the source, make that clear.

Help the student understand the
topic accurately and clearly."""
                                ),
                                (
                                    "human",
                                    """Source Context:

{context}

Question:

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

                    response = llm.invoke(
                        final_prompt
                    )

                    answer = response.content

            processing.empty()

            st.write(
                answer
            )

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )

        except Exception as e:

            processing.empty()

            answer = (
                "Sorry, something went wrong "
                "while processing your request."
            )

            st.error(
                answer
            )

            st.caption(
                str(e)
)
