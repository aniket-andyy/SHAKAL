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
    #MainMenu, footer, header {
        visibility: hidden;
    }

    .block-container {
        max-width: 1050px;
        padding-top: 2rem;
        padding-bottom: 5rem;
    }

    .hero {
        text-align: center;
        padding: 20px 0 35px 0;
    }

    .brand {
        font-size: 56px;
        font-weight: 900;
        letter-spacing: 8px;
        line-height: 1;
    }

    .subtitle {
        font-size: 18px;
        font-weight: 600;
        letter-spacing: 7px;
        opacity: 0.65;
        margin-top: 10px;
    }

    .tagline {
        font-size: 15px;
        opacity: 0.55;
        margin-top: 18px;
    }

    .model {
        font-size: 13px;
        opacity: 0.45;
        margin-top: 7px;
    }

    .developer {
        font-size: 13px;
        opacity: 0.55;
        margin-top: 16px;
    }

    .developer a {
        text-decoration: none;
    }

    .section-title {
        font-size: 24px;
        font-weight: 750;
        margin-top: 30px;
        margin-bottom: 5px;
    }

    .section-description {
        font-size: 14px;
        opacity: 0.55;
        margin-bottom: 20px;
    }

    .card {
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 16px;
        padding: 20px;
        background: rgba(255,255,255,0.025);
        margin-bottom: 15px;
    }

    .source-limit {
        text-align: right;
        font-size: 12px;
        opacity: 0.5;
        margin-bottom: 10px;
    }

    .source-ready {
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 14px;
        padding: 16px;
        margin-top: 20px;
        background: rgba(255,255,255,0.025);
    }

    .processing {
        text-align: center;
        padding: 15px;
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 14px;
        margin-bottom: 15px;
    }

    .processing-main {
        font-weight: 700;
        font-size: 16px;
    }

    .processing-message {
        font-size: 12px;
        opacity: 0.5;
        margin-top: 4px;
    }

    .personality-info {
        font-size: 13px;
        opacity: 0.55;
        margin-top: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


if "sources" not in st.session_state:
    st.session_state.sources = []

if "source_ready" not in st.session_state:
    st.session_state.source_ready = False

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "personality" not in st.session_state:
    st.session_state.personality = "Normal"


PERSONALITIES = {
    "Normal": """You are SHAKAL, a balanced AI study assistant.

Help students and learners understand subjects clearly.

Explain concepts accurately and at an appropriate level of detail.
Use examples, analogies, step-by-step explanations, and concise
summaries when useful.

Adapt your teaching style to the user's question.
Do not unnecessarily overcomplicate simple questions.""",

    "Theorist": """You are SHAKAL in Theorist personality.

Focus on the underlying theory and conceptual foundations.

Explain definitions, principles, relationships between concepts,
assumptions, mathematical reasoning, cause and effect, and why
something works.

Prefer deep conceptual understanding over quick answers.

When appropriate, connect the current concept with related
theories and fundamentals.""",

    "Practicalist": """You are SHAKAL in Practicalist personality.

Focus on how knowledge is used in the real world.

Explain practical applications, implementation, real-world examples,
workflows, demonstrations, use cases, and common mistakes.

Whenever useful, convert theory into something the learner can
actually do or implement.""",

    "Examiner": """You are SHAKAL in Examiner personality.

Act like a strict but helpful academic examiner.

Focus on important concepts, likely exam questions, conceptual gaps,
mistakes, problem-solving, and evaluation of answers.

When the user provides an answer, evaluate it, identify mistakes,
explain why they are mistakes, and show how to improve.

Do not praise unnecessarily. Focus on useful feedback.""",

    "Guide": """You are SHAKAL in Guide personality.

Act as a supportive study guide who helps the learner move from
confusion to understanding.

Break difficult subjects into manageable steps.

Help the student decide what to learn first, what to practice next,
what to revise, and how to approach difficult problems.

Ask useful guiding questions when appropriate.

Give clear study direction without doing all the thinking for the
student.

The goal is to help the learner become more independent and
confident in studying."""
}


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
    '<div class="section-description">'
    'Add up to 3 study sources to build your knowledge base.'
    '</div>',
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
    index=0
)


pdf_file = None
website_url = ""
youtube_url = ""
image_file = None


if source_type == "📄 PDF":

    pdf_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        accept_multiple_files=False
    )


elif source_type == "🌐 Website":

    website_url = st.text_input(
        "Website URL",
        placeholder="https://example.com"
    )


elif source_type == "🎥 YouTube":

    youtube_url = st.text_input(
        "YouTube Video URL",
        placeholder="https://youtube.com/watch?v=..."
    )


elif source_type == "🖼️ Image":

    image_file = st.file_uploader(
        "Upload Image",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=False
    )


st.markdown(
    f'<div class="source-limit">'
    f'{len(st.session_state.sources)} / 3 sources added'
    f'</div>',
    unsafe_allow_html=True
)


add_source = st.button(
    "＋ Add Source",
    use_container_width=True
)


if add_source:

    if len(st.session_state.sources) >= 3:

        st.error("Maximum 3 sources are allowed at a time.")

    else:

        valid = False
        source_label = ""

        if source_type == "📄 PDF" and pdf_file:

            valid = True
            source_label = f"📄 {pdf_file.name}"

        elif source_type == "🌐 Website" and website_url.strip():

            valid = True
            source_label = f"🌐 {website_url.strip()}"

        elif source_type == "🎥 YouTube" and youtube_url.strip():

            valid = True
            source_label = "🎥 YouTube Video"

        elif source_type == "🖼️ Image" and image_file:

            valid = True
            source_label = f"🖼️ {image_file.name}"

        if not valid:

            st.warning("Please provide a valid source first.")

        else:

            st.session_state.sources.append(
                {
                    "type": source_type,
                    "label": source_label,
                    "file": pdf_file if source_type == "📄 PDF" else (
                        image_file if source_type == "🖼️ Image" else None
                    ),
                    "url": website_url.strip() if source_type == "🌐 Website" else (
                        youtube_url.strip() if source_type == "🎥 YouTube" else ""
                    )
                }
            )

            st.rerun()


if st.session_state.sources:

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    for index, source in enumerate(
        st.session_state.sources,
        start=1
    ):

        col1, col2 = st.columns(
            [8, 1]
        )

        with col1:

            st.write(
                f"**{index}. {source['label']}**"
            )

        with col2:

            if st.button(
                "×",
                key=f"remove_source_{index}"
            ):

                st.session_state.sources.pop(
                    index - 1
                )

                st.session_state.source_ready = False

                st.rerun()

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


process_sources = st.button(
    "⚡ Process Sources",
    use_container_width=True,
    type="primary"
)


if process_sources:

    if not st.session_state.sources:

        st.warning("Add at least one source.")

    else:

        all_docs = []

        progress = st.progress(0)

        status = st.empty()

        try:

            total = len(
                st.session_state.sources
            )

            for index, source in enumerate(
                st.session_state.sources,
                start=1
            ):

                status.info(
                    f"Processing source {index}/{total}..."
                )

                source_type_value = source["type"]

                if source_type_value == "📄 PDF":

                    uploaded = source["file"]

                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".pdf"
                    ) as temp:

                        temp.write(
                            uploaded.getbuffer()
                        )

                        temp_path = temp.name

                    try:

                        docs = load_pdf(
                            temp_path
                        )

                        for doc in docs:

                            doc.metadata["source_type"] = "pdf"
                            doc.metadata["source"] = uploaded.name

                        all_docs.extend(docs)

                    finally:

                        if os.path.exists(temp_path):
                            os.remove(temp_path)

                elif source_type_value == "🌐 Website":

                    docs = load_webpage(
                        source["url"]
                    )

                    for doc in docs:

                        doc.metadata["source_type"] = "webpage"
                        doc.metadata["source"] = source["url"]

                    all_docs.extend(docs)

                elif source_type_value == "🎥 YouTube":

                    docs = load_youtube(
                        source["url"]
                    )

                    all_docs.extend(docs)

                elif source_type_value == "🖼️ Image":

                    uploaded = source["file"]

                    extension = os.path.splitext(
                        uploaded.name
                    )[1]

                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=extension
                    ) as temp:

                        temp.write(
                            uploaded.getbuffer()
                        )

                        temp_path = temp.name

                    try:

                        docs = load_image(
                            temp_path
                        )

                        for doc in docs:

                            doc.metadata["source_type"] = "image"
                            doc.metadata["source"] = uploaded.name

                        all_docs.extend(docs)

                    finally:

                        if os.path.exists(temp_path):
                            os.remove(temp_path)

                progress.progress(
                    index / total
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

            status.empty()

            st.error(
                "Unable to process the selected sources. "
                "Please check the files or URLs and try again."
            )


if st.session_state.source_ready:

    st.markdown(
        """
        <div class="source-ready">
            <strong>✓ SOURCE READY</strong><br>
            <span style="opacity:0.55;">
                Your selected study material is available to SHAKAL.
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
    '<div class="section-description">'
    'Choose how SHAKAL should teach you.'
    '</div>',
    unsafe_allow_html=True
)


personality = st.selectbox(
    "SHAKAL personality",
    list(PERSONALITIES.keys()),
    index=0
)

st.session_state.personality = personality


personality_descriptions = {
    "Normal": "Balanced explanations for everyday studying.",
    "Theorist": "Deep focus on theory, concepts and fundamentals.",
    "Practicalist": "Focus on real-world applications and implementation.",
    "Examiner": "Strict academic evaluation and exam-focused learning.",
    "Guide": "Step-by-step guidance toward independent learning."
}


st.markdown(
    f'<div class="personality-info">'
    f'{personality_descriptions[personality]}'
    f'</div>',
    unsafe_allow_html=True
)


st.markdown(
    '<div class="section-title">Study Chat</div>',
    unsafe_allow_html=True
)


for message in st.session_state.chat_history:

    with st.chat_message(
        message["role"]
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

    with st.chat_message("user"):

        st.write(query)

    with st.chat_message("assistant"):

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

            llm = ChatMistralAI(
                model="mistral-small-2506"
            )

            personality_prompt = PERSONALITIES[
                st.session_state.personality
            ]

            if st.session_state.source_ready:

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

The student's provided study sources are the PRIMARY source
for answering the question.

Use the source whenever relevant.

If the source does not contain enough information, you may use
your general knowledge to help the student.

Clearly distinguish information from the source from information
added using general knowledge.

Never invent information.
"""

                prompt = ChatPromptTemplate.from_messages(
                    [
                        (
                            "system",
                            system_prompt
                        ),
                        (
                            "human",
                            """Study Source Context:

{context}

Student Question:

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
                            personality_prompt
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

            processing.empty()

            st.write(answer)

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )

        except Exception:

            processing.empty()

            st.error(
                "Sorry, SHAKAL could not process your request."
)
