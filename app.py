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
    layout="centered"
)


if "MISTRAL_API_KEY" in st.secrets:
    os.environ["MISTRAL_API_KEY"] = st.secrets["MISTRAL_API_KEY"]

if "mode" not in st.session_state:
    st.session_state.mode = "Source + AI"

if "personality" not in st.session_state:
    st.session_state.personality = "Normal"

if "sources" not in st.session_state:
    st.session_state.sources = []

if "source_ready" not in st.session_state:
    st.session_state.source_ready = False

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


PERSONALITIES = {
    "Normal": """
You are SHAKAL, a balanced AI study assistant.

Help students and learners understand subjects clearly.

Explain concepts accurately and at an appropriate level of
detail. Use examples, analogies, step-by-step explanations,
and concise summaries when useful.

Adapt your teaching style to the user's question.
Do not unnecessarily overcomplicate simple questions.
""",

    "Theorist": """
You are SHAKAL in Theorist personality.

Focus on the underlying theory and conceptual foundations.

Explain:
- definitions
- principles
- relationships between concepts
- assumptions
- mathematical reasoning
- cause and effect
- why something works

Prefer deep conceptual understanding over quick answers.

When appropriate, connect the current concept with related
theories and fundamentals.
""",

    "Practicalist": """
You are SHAKAL in Practicalist personality.

Focus on how knowledge is used in the real world.

Explain:
- practical applications
- implementation
- real-world examples
- workflows
- demonstrations
- use cases
- common mistakes

Whenever useful, convert theory into something the learner
can actually do or implement.
""",

    "Examiner": """
You are SHAKAL in Examiner personality.

Act like a strict but helpful academic examiner.

Focus on:
- important concepts
- likely exam questions
- conceptual gaps
- mistakes
- problem-solving
- evaluation of answers

When the user provides an answer, evaluate it, identify
mistakes, explain why they are mistakes, and show how to
improve.

Do not praise unnecessarily. Focus on useful feedback.
""",

    "Guide": """
You are SHAKAL in Guide personality.

Act as a personal study guide who helps the learner move
from confusion to understanding step by step.

Identify what the student already understands and what
they need to learn next.

Break difficult topics into manageable steps.

When useful:
- create a learning path
- suggest what to study next
- explain prerequisites
- provide practice tasks
- ask useful guiding questions
- help the student revise
- connect related concepts

Do not overwhelm the student with unnecessary information.

Your goal is to guide the learner toward independent
understanding rather than simply giving answers.
"""
}


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
        max-width: 950px;
        padding-top: 2rem;
        padding-bottom: 5rem;
    }

    .hero {
        text-align: center;
        padding: 35px 10px 25px 10px;
    }

    .brand {
        font-size: 54px;
        font-weight: 900;
        letter-spacing: 8px;
        line-height: 1;
    }

    .subtitle {
        margin-top: 10px;
        font-size: 18px;
        font-weight: 600;
        letter-spacing: 7px;
        opacity: 0.55;
    }

    .tagline {
        margin-top: 18px;
        font-size: 15px;
        opacity: 0.6;
    }

    .model {
        margin-top: 8px;
        font-size: 12px;
        opacity: 0.45;
    }

    .developer {
        margin-top: 14px;
        font-size: 12px;
        opacity: 0.5;
    }

    .developer a {
        text-decoration: none;
    }

    .section {
        margin-top: 35px;
    }

    .section-title {
        font-size: 21px;
        font-weight: 750;
        margin-bottom: 4px;
    }

    .section-subtitle {
        font-size: 13px;
        opacity: 0.5;
        margin-bottom: 18px;
    }

    .source-limit {
        text-align: right;
        font-size: 12px;
        opacity: 0.45;
        margin-bottom: 8px;
    }

    .source-ready {
        padding: 14px 16px;
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 12px;
        margin-top: 12px;
        background: rgba(255,255,255,0.025);
    }

    .processing {
        text-align: center;
        padding: 18px;
        margin: 12px 0;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
    }

    .processing-title {
        font-weight: 700;
        font-size: 15px;
    }

    .processing-text {
        font-size: 12px;
        opacity: 0.45;
        margin-top: 5px;
    }

    .personality-card {
        padding: 16px;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.08);
        background: rgba(255,255,255,0.025);
    }

    </style>
    """,
    unsafe_allow_html=True
)


st.markdown(
    """
    <div class="hero">
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
            <a href="https://www.linkedin.com/in/aniket-sharma-42a700418?utm_source=share_via&utm_content=member_android" target="_blank">
                LinkedIn
            </a>
            &nbsp;&nbsp;·&nbsp;&nbsp;
            <a href="https://github.com/aniket-andyy" target="_blank">
                GitHub
            </a>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


st.markdown(
    '<div class="section-title">Your Sources</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">'
    'Add study material for SHAKAL to learn from.'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="source-limit">Maximum 3 sources at a time</div>',
    unsafe_allow_html=True
)


source_type = st.selectbox(
    "Source type",
    [
        "📄 PDF",
        "🌐 Website",
        "🎥 YouTube",
        "🖼️ Image"
    ],
    label_visibility="collapsed"
)


selected_file = None
selected_url = None


if source_type == "📄 PDF":

    selected_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        accept_multiple_files=False
    )


elif source_type == "🌐 Website":

    selected_url = st.text_input(
        "Website URL",
        placeholder="https://example.com"
    )


elif source_type == "🎥 YouTube":

    selected_url = st.text_input(
        "YouTube URL",
        placeholder="https://youtube.com/watch?v=..."
    )


elif source_type == "🖼️ Image":

    selected_file = st.file_uploader(
        "Upload image",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=False
    )


add_source = st.button(
    "＋ Add Source",
    use_container_width=True
)


if add_source:

    if len(st.session_state.sources) >= 3:

        st.error(
            "Maximum 3 sources allowed at a time."
        )

    elif source_type == "📄 PDF" and selected_file is None:

        st.warning(
            "Please select a PDF."
        )

    elif source_type == "🖼️ Image" and selected_file is None:

        st.warning(
            "Please select an image."
        )

    elif source_type in ["🌐 Website", "🎥 YouTube"] and not selected_url:

        st.warning(
            "Please enter a URL."
        )

    else:

        if source_type == "📄 PDF":

            st.session_state.sources.append(
                {
                    "type": "pdf",
                    "name": selected_file.name,
                    "file": selected_file
                }
            )

        elif source_type == "🖼️ Image":

            st.session_state.sources.append(
                {
                    "type": "image",
                    "name": selected_file.name,
                    "file": selected_file
                }
            )

        elif source_type == "🌐 Website":

            st.session_state.sources.append(
                {
                    "type": "website",
                    "name": selected_url,
                    "url": selected_url
                }
            )

        elif source_type == "🎥 YouTube":

            st.session_state.sources.append(
                {
                    "type": "youtube",
                    "name": selected_url,
                    "url": selected_url
                }
            )

        st.rerun()


if st.session_state.sources:

    st.markdown(
        f"**{len(st.session_state.sources)}/3 sources added**"
    )

    for index, source in enumerate(
        st.session_state.sources
    ):

        col1, col2 = st.columns(
            [8, 1]
        )

        with col1:

            if source["type"] == "pdf":
                icon = "📄"

            elif source["type"] == "website":
                icon = "🌐"

            elif source["type"] == "youtube":
                icon = "🎥"

            else:
                icon = "🖼️"

            st.caption(
                f"{icon} {source['name']}"
            )

        with col2:

            if st.button(
                "×",
                key=f"remove_source_{index}"
            ):

                st.session_state.sources.pop(
                    index
                )

                st.rerun()


process_sources = st.button(
    "Process Sources",
    use_container_width=True,
    type="primary",
    disabled=len(st.session_state.sources) == 0
)


if process_sources:

    all_docs = []

    progress = st.progress(0)
    status = st.empty()

    total = len(
        st.session_state.sources
    )

    try:

        for index, source in enumerate(
            st.session_state.sources
        ):

            if source["type"] == "pdf":

                status.info(
                    f"Processing PDF · {source['name']}"
                )

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as temp:

                    temp.write(
                        source["file"].getbuffer()
                    )

                    temp_path = temp.name

                try:

                    docs = load_pdf(
                        temp_path
                    )

                finally:

                    if os.path.exists(temp_path):
                        os.remove(temp_path)

                for doc in docs:

                    doc.metadata["source_type"] = "pdf"
                    doc.metadata["source"] = source["name"]

            elif source["type"] == "website":

                status.info(
                    "Processing website..."
                )

                docs = load_webpage(
                    source["url"]
                )

                for doc in docs:

                    doc.metadata["source_type"] = "webpage"
                    doc.metadata["source"] = source["url"]

            elif source["type"] == "youtube":

                status.info(
                    "Extracting YouTube transcript..."
                )

                docs = load_youtube(
                    source["url"]
                )

            else:

                status.info(
                    f"Running OCR · {source['name']}"
                )

                extension = os.path.splitext(
                    source["file"].name
                )[1]

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=extension
                ) as temp:

                    temp.write(
                        source["file"].getbuffer()
                    )

                    temp_path = temp.name

                try:

                    docs = load_image(
                        temp_path
                    )

                finally:

                    if os.path.exists(temp_path):
                        os.remove(temp_path)

                for doc in docs:

                    doc.metadata["source_type"] = "image"
                    doc.metadata["source"] = source["name"]

            all_docs.extend(docs)

            progress.progress(
                (index + 1) / total
            )

        status.info(
            "Creating embeddings and updating knowledge base..."
        )

        add_to_chroma(
            all_docs
        )

        progress.progress(1.0)

        status.success(
            "Sources processed successfully."
        )

        st.session_state.source_ready = True

    except Exception:

        progress.empty()

        status.error(
            "Unable to process the selected sources. "
            "Please check the files or URLs and try again."
        )


if st.session_state.source_ready:

    st.markdown(
        """
        <div class="source-ready">
            <b>✓ KNOWLEDGE BASE READY</b>
            <br>
            <span style="opacity:0.55;font-size:12px;">
                SHAKAL can now use your sources.
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )


st.markdown(
    '<div class="section-title">Personality</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">'
    'Choose how SHAKAL should teach you.'
    '</div>',
    unsafe_allow_html=True
)


personality = st.selectbox(
    "Choose personality",
    [
        "Normal",
        "Theorist",
        "Practicalist",
        "Examiner",
        "Guide"
    ],
    index=0,
    key="personality",
    label_visibility="collapsed"
)


PERSONALITY_DESCRIPTIONS = {
    "Normal": "Balanced explanations with examples and clear summaries.",
    "Theorist": "Deep focus on theory, principles and conceptual foundations.",
    "Practicalist": "Focuses on implementation, applications and real-world use.",
    "Examiner": "Strict academic evaluation, mistakes and exam preparation.",
    "Guide": "Guides you step-by-step toward understanding and independent learning."
}


st.caption(
    PERSONALITY_DESCRIPTIONS[personality]
)


st.markdown(
    '<div class="section-title">Study Chat</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">'
    'Ask questions, learn concepts, revise and practice.'
    '</div>',
    unsafe_allow_html=True
)


for message in st.session_state.chat_history:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
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

    with st.chat_message("user"):

        st.markdown(query)

    with st.chat_message("assistant"):

        processing = st.empty()

        processing.markdown(
            """
            <div class="processing">
                <div class="processing-title">
                    Processing
                </div>
                <div class="processing-text">
                    [ aniket bhai ki taraf se hello! :) ]
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        try:

            llm = ChatMistralAI(
                model="mistral-small-2506"
            )

            personality_prompt = PERSONALITIES[
                personality
            ]

            if st.session_state.source_ready:

                embedding_model = HuggingFaceEmbeddings(
                    model_name=(
                        "sentence-transformers/"
                        "all-MiniLM-L6-v2"
                    )
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

                docs = retriever.invoke(
                    query
                )

                context = "\n\n".join(
                    doc.page_content
                    for doc in docs
                )

                system_prompt = f"""
{personality_prompt}

You are also a source-grounded study assistant.

The user has provided study material.

Use the provided source as the PRIMARY source for answering.

If the source does not contain enough information,
you may use your general knowledge.

If you use information that is not directly present
in the source, clearly indicate that it comes from
general knowledge.

Never invent information.

Study Source Context:

{{context}}
"""

                prompt = ChatPromptTemplate.from_messages(
                    [
                        (
                            "system",
                            system_prompt
                        ),
                        (
                            "human",
                            """Student's Question:

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

            else:

                prompt = ChatPromptTemplate.from_messages(
                    [
                        (
                            "system",
                            f"""
{personality_prompt}

You are SHAKAL, a study AI assistant.

Help students and learners understand,
learn, revise and practice academic topics.

There is currently no external study source.

Use your general knowledge.

Be accurate, clear and student-friendly.
"""
                        ),
                        (
                            "human",
                            """Student's Question:

{question}"""
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

            processing.empty()

            st.markdown(answer)

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )

        except Exception:

            processing.empty()

            error_message = (
                "Sorry, SHAKAL couldn't process "
                "your request right now."
            )

            st.error(
                error_message
            )
