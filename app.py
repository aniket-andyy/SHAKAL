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


st.markdown(
    """
    <style>
    #MainMenu,
    footer,
    header {
        visibility: hidden;
    }

    .block-container {
        max-width: 900px;
        padding-top: 35px;
        padding-bottom: 80px;
    }

    .brand {
        text-align: center;
        font-size: 52px;
        font-weight: 900;
        letter-spacing: 7px;
        line-height: 1;
    }

    .subtitle {
        text-align: center;
        font-size: 17px;
        font-weight: 600;
        letter-spacing: 6px;
        opacity: 0.55;
        margin-top: 10px;
    }

    .tagline {
        text-align: center;
        font-size: 15px;
        opacity: 0.55;
        margin-top: 20px;
    }

    .model {
        text-align: center;
        font-size: 13px;
        opacity: 0.45;
        margin-top: 7px;
    }

    .developer {
        text-align: center;
        font-size: 13px;
        opacity: 0.55;
        margin-top: 15px;
        line-height: 1.6;
    }

    .developer a {
        text-decoration: none;
    }

    .section {
        margin-top: 45px;
    }

    .section-title {
        font-size: 22px;
        font-weight: 750;
        margin-bottom: 5px;
    }

    .section-description {
        font-size: 14px;
        opacity: 0.55;
        margin-bottom: 20px;
    }

    .source-note {
        font-size: 12px;
        opacity: 0.5;
        margin-top: 8px;
        margin-bottom: 18px;
    }

    .source-item {
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 10px;
        padding: 12px 15px;
        margin-bottom: 8px;
        font-size: 14px;
    }

    .processing {
        text-align: center;
        padding: 15px;
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 10px;
        margin: 15px 0;
    }

    .processing-main {
        font-size: 15px;
        font-weight: 700;
    }

    .processing-message {
        font-size: 12px;
        opacity: 0.5;
        margin-top: 4px;
    }

    .ready {
        margin-top: 15px;
        padding: 12px 15px;
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.10);
        font-size: 13px;
    }

    .personality-description {
        font-size: 13px;
        opacity: 0.5;
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
    """,
    unsafe_allow_html=True
)


# ============================================================
# SOURCES
# ============================================================

st.markdown(
    '<div class="section">',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-title">Upload Your Sources</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'Give SHAKAL your study material and start learning.'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    f'<div class="source-note">'
    f'You can add up to 3 sources at a time · '
    f'{len(st.session_state.sources)}/3 sources added'
    '</div>',
    unsafe_allow_html=True
)


# ------------------------------------------------------------
# PDF
# ------------------------------------------------------------

pdf_file = st.file_uploader(
    "📄 PDF",
    type=["pdf"],
    accept_multiple_files=False
)

if pdf_file:

    if not any(
        source["id"] == f"pdf_{pdf_file.name}"
        for source in st.session_state.sources
    ):

        if len(st.session_state.sources) < 3:

            st.session_state.sources.append(
                {
                    "id": f"pdf_{pdf_file.name}",
                    "type": "pdf",
                    "name": pdf_file.name,
                    "file": pdf_file,
                    "url": None
                }
            )

        else:

            st.warning(
                "Maximum 3 sources allowed."
            )


# ------------------------------------------------------------
# WEBSITE
# ------------------------------------------------------------

website_url = st.text_input(
    "🌐 Website",
    placeholder="Paste website URL"
)

if st.button(
    "Add Website",
    use_container_width=True
):

    if not website_url.strip():

        st.warning(
            "Enter a website URL first."
        )

    elif len(st.session_state.sources) >= 3:

        st.warning(
            "Maximum 3 sources allowed."
        )

    else:

        st.session_state.sources.append(
            {
                "id": f"web_{website_url.strip()}",
                "type": "web",
                "name": website_url.strip(),
                "file": None,
                "url": website_url.strip()
            }
        )

        st.rerun()


# ------------------------------------------------------------
# YOUTUBE
# ------------------------------------------------------------

youtube_url = st.text_input(
    "🎥 YouTube Video",
    placeholder="Paste YouTube video URL"
)

if st.button(
    "Add YouTube",
    use_container_width=True
):

    if not youtube_url.strip():

        st.warning(
            "Enter a YouTube URL first."
        )

    elif len(st.session_state.sources) >= 3:

        st.warning(
            "Maximum 3 sources allowed."
        )

    else:

        st.session_state.sources.append(
            {
                "id": f"yt_{youtube_url.strip()}",
                "type": "youtube",
                "name": youtube_url.strip(),
                "file": None,
                "url": youtube_url.strip()
            }
        )

        st.rerun()


# ------------------------------------------------------------
# IMAGE
# ------------------------------------------------------------

image_file = st.file_uploader(
    "🖼️ Image",
    type=["png", "jpg", "jpeg", "webp"],
    accept_multiple_files=False
)

if image_file:

    if not any(
        source["id"] == f"image_{image_file.name}"
        for source in st.session_state.sources
    ):

        if len(st.session_state.sources) < 3:

            st.session_state.sources.append(
                {
                    "id": f"image_{image_file.name}",
                    "type": "image",
                    "name": image_file.name,
                    "file": image_file,
                    "url": None
                }
            )

        else:

            st.warning(
                "Maximum 3 sources allowed."
            )


# ============================================================
# SELECTED SOURCES
# ============================================================

if st.session_state.sources:

    st.markdown(
        '<div class="section-title">Selected Sources</div>',
        unsafe_allow_html=True
    )

    for index, source in enumerate(
        st.session_state.sources
    ):

        col1, col2 = st.columns(
            [9, 1]
        )

        with col1:

            icons = {
                "pdf": "📄",
                "web": "🌐",
                "youtube": "🎥",
                "image": "🖼️"
            }

            st.markdown(
                f"""
                <div class="source-item">
                    {icons[source["type"]]}&nbsp;&nbsp;
                    {source["name"]}
                </div>
                """,
                unsafe_allow_html=True
            )

        with col2:

            if st.button(
                "×",
                key=f"remove_{source['id']}"
            ):

                st.session_state.sources.pop(
                    index
                )

                st.session_state.source_ready = False

                st.rerun()


# ============================================================
# PROCESS
# ============================================================

if st.session_state.sources:

    if st.button(
        "⚡ Process Sources",
        use_container_width=True,
        type="primary"
    ):

        all_docs = []

        progress = st.progress(
            0
        )

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

                if source["type"] == "pdf":

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

                        for doc in docs:

                            doc.metadata[
                                "source_type"
                            ] = "pdf"

                            doc.metadata[
                                "source"
                            ] = source["name"]

                        all_docs.extend(
                            docs
                        )

                    finally:

                        if os.path.exists(
                            temp_path
                        ):

                            os.remove(
                                temp_path
                            )

                elif source["type"] == "web":

                    docs = load_webpage(
                        source["url"]
                    )

                    for doc in docs:

                        doc.metadata[
                            "source_type"
                        ] = "webpage"

                        doc.metadata[
                            "source"
                        ] = source["url"]

                    all_docs.extend(
                        docs
                    )

                elif source["type"] == "youtube":

                    docs = load_youtube(
                        source["url"]
                    )

                    all_docs.extend(
                        docs
                    )

                elif source["type"] == "image":

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

                        for doc in docs:

                            doc.metadata[
                                "source_type"
                            ] = "image"

                            doc.metadata[
                                "source"
                            ] = source["name"]

                        all_docs.extend(
                            docs
                        )

                    finally:

                        if os.path.exists(
                            temp_path
                        ):

                            os.remove(
                                temp_path
                            )

                progress.progress(
                    index / total
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
                "Sources are ready."
            )

            st.session_state.source_ready = True

        except Exception:

            status.empty()

            st.error(
                "Unable to process the selected sources."
            )


if st.session_state.source_ready:

    st.markdown(
        """
        <div class="ready">
            ✓ SOURCE READY
            <br>
            <span style="opacity:0.5;">
                Your study material is available to SHAKAL.
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# PERSONALITY
# ============================================================

st.markdown(
    '<div class="section">',
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
    "SHAKAL Personality",
    [
        "Normal",
        "Theorist",
        "Practicalist",
        "Examiner",
        "Guide"
    ],
    index=0
)

st.session_state.personality = personality


descriptions = {
    "Normal": "Balanced explanations for everyday studying.",
    "Theorist": "Deep focus on theory, concepts and fundamentals.",
    "Practicalist": "Focus on real-world applications and implementation.",
    "Examiner": "Strict academic evaluation and exam-focused learning.",
    "Guide": "Step-by-step guidance toward independent learning."
}


st.markdown(
    f'<div class="personality-description">'
    f'{descriptions[personality]}'
    f'</div>',
    unsafe_allow_html=True
)


# ============================================================
# CHAT
# ============================================================

st.markdown(
    '<div class="section">',
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

            llm = ChatMistralAI(
                model="mistral-small-2506"
            )

            personality_prompt = PERSONALITIES[
                st.session_state.personality
            ]

            if not st.session_state.source_ready:

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

            else:

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

You are SHAKAL, a study AI assistant.

The user's uploaded study sources are the PRIMARY source
for answering questions.

Use the provided source whenever relevant.

You may use general knowledge when the source does not
contain enough information.

If you use general knowledge, clearly indicate that it is
not directly present in the provided source.

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
                            """Study Source:

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

        except Exception:

            processing.empty()

            st.error(
                "Sorry, SHAKAL could not process your request."
                    )
