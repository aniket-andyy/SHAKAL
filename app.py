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
    layout="wide"
)

st.markdown(
    """
    <style>
    #MainMenu, footer, header {
        visibility: hidden;
    }

    .block-container {
        max-width: 1000px;
        padding-top: 3rem;
        padding-bottom: 4rem;
    }

    .brand {
        text-align: center;
        font-size: 56px;
        font-weight: 900;
        letter-spacing: 8px;
        line-height: 1;
    }

    .subtitle {
        text-align: center;
        font-size: 20px;
        font-weight: 600;
        letter-spacing: 7px;
        opacity: 0.65;
        margin-top: 8px;
    }

    .tagline {
        text-align: center;
        font-size: 15px;
        opacity: 0.6;
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
        margin-top: 14px;
    }

    .developer a {
        text-decoration: none;
    }

    .section-title {
        font-size: 23px;
        font-weight: 750;
        margin-top: 45px;
        margin-bottom: 5px;
    }

    .section-subtitle {
        font-size: 14px;
        opacity: 0.55;
        margin-bottom: 20px;
    }

    .source-note {
        padding: 12px 16px;
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.10);
        background: rgba(255,255,255,0.025);
        font-size: 13px;
        opacity: 0.7;
        margin-bottom: 18px;
    }

    .processing {
        text-align: center;
        padding: 18px;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.10);
        margin: 15px 0;
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

    .source-ready {
        padding: 14px 17px;
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.10);
        margin-top: 15px;
    }

    .source-ready-title {
        font-weight: 700;
        font-size: 13px;
    }

    .source-ready-text {
        font-size: 12px;
        opacity: 0.55;
        margin-top: 4px;
    }

    .personality-info {
        text-align: center;
        font-size: 13px;
        opacity: 0.5;
        margin-top: 10px;
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


st.markdown(
    '<div class="section-title">Upload Your Sources</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">'
    'Give SHAKAL your study material.'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="source-note">'
    'You can add up to <b>3 sources</b> at a time.'
    '</div>',
    unsafe_allow_html=True
)


source_type = st.radio(
    "Source type",
    [
        "📄 PDF",
        "🌐 Website",
        "🎥 YouTube",
        "🖼️ Image"
    ],
    horizontal=True,
    label_visibility="collapsed"
)


pdf_files = []
website_urls = []
youtube_urls = []
image_files = []


if source_type == "📄 PDF":

    pdf_files = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        accept_multiple_files=True
    )

elif source_type == "🌐 Website":

    website_text = st.text_area(
        "Website URLs",
        placeholder="Paste one URL per line..."
    )

    website_urls = [
        x.strip()
        for x in website_text.splitlines()
        if x.strip()
    ]

elif source_type == "🎥 YouTube":

    youtube_text = st.text_area(
        "YouTube URLs",
        placeholder="Paste one YouTube URL per line..."
    )

    youtube_urls = [
        x.strip()
        for x in youtube_text.splitlines()
        if x.strip()
    ]

else:

    image_files = st.file_uploader(
        "Upload images",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True
    )


total_sources = (
    len(pdf_files or [])
    + len(website_urls)
    + len(youtube_urls)
    + len(image_files or [])
)


st.caption(f"{total_sources}/3 sources selected")


process_button = st.button(
    "Process Sources",
    use_container_width=True,
    type="primary"
)


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

        try:

            current = 0

            if pdf_files:

                for file in pdf_files:

                    status.info(
                        f"Processing {file.name}..."
                    )

                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".pdf"
                    ) as temp:

                        temp.write(file.getbuffer())
                        temp_path = temp.name

                    try:

                        docs = load_pdf(temp_path)

                        for doc in docs:
                            doc.metadata["source_type"] = "pdf"
                            doc.metadata["source"] = file.name

                        all_docs.extend(docs)
                        source_names.append(
                            f"📄 {file.name}"
                        )

                    finally:

                        if os.path.exists(temp_path):
                            os.remove(temp_path)

                    current += 1
                    progress.progress(current / total_sources)

            for url in website_urls:

                status.info(
                    f"Processing website..."
                )

                docs = load_webpage(url)

                for doc in docs:
                    doc.metadata["source_type"] = "webpage"
                    doc.metadata["source"] = url

                all_docs.extend(docs)
                source_names.append(
                    f"🌐 {url}"
                )

                current += 1
                progress.progress(current / total_sources)

            for url in youtube_urls:

                status.info(
                    "Processing YouTube transcript..."
                )

                docs = load_youtube(url)

                all_docs.extend(docs)
                source_names.append(
                    f"🎥 YouTube"
                )

                current += 1
                progress.progress(current / total_sources)

            if image_files:

                for file in image_files:

                    status.info(
                        f"Processing {file.name}..."
                    )

                    extension = os.path.splitext(
                        file.name
                    )[1]

                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=extension
                    ) as temp:

                        temp.write(file.getbuffer())
                        temp_path = temp.name

                    try:

                        docs = load_image(temp_path)

                        for doc in docs:
                            doc.metadata["source_type"] = "image"
                            doc.metadata["source"] = file.name

                        all_docs.extend(docs)
                        source_names.append(
                            f"🖼️ {file.name}"
                        )

                    finally:

                        if os.path.exists(temp_path):
                            os.remove(temp_path)

                    current += 1
                    progress.progress(current / total_sources)

            status.info(
                "Creating embeddings and updating knowledge base..."
            )

            add_to_chroma(all_docs)

            progress.progress(1.0)

            status.success(
                "Sources ready."
            )

            st.session_state.sources = source_names
            st.session_state.source_ready = True

        except Exception:

            status.empty()

            st.error(
                "Unable to process the selected source."
            )


if st.session_state.source_ready:

    st.markdown(
        """
        <div class="source-ready">
            <div class="source-ready-title">
                ✓ SOURCE READY
            </div>
            <div class="source-ready-text">
                Your study material is available to SHAKAL.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    for source in st.session_state.sources:
        st.caption(source)


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


personality_columns = st.columns(5)

personalities = [
    "Normal",
    "Theorist",
    "Practicalist",
    "Examiner",
    "Guide"
]


for column, personality in zip(
    personality_columns,
    personalities
):

    with column:

        if st.button(
            personality,
            use_container_width=True,
            type=(
                "primary"
                if st.session_state.personality == personality
                else "secondary"
            ),
            key=f"personality_{personality}"
        ):

            st.session_state.personality = personality
            st.rerun()


st.markdown(
    f"""
    <div class="personality-info">
        Active personality · {st.session_state.personality}
    </div>
    """,
    unsafe_allow_html=True
)


st.markdown(
    '<div class="section-title">Study Chat</div>',
    unsafe_allow_html=True
)


reset_col, status_col = st.columns([1, 4])

with reset_col:

    if st.button(
        "Reset Chat",
        use_container_width=True
    ):

        st.session_state.chat_history = []
        st.rerun()


with status_col:

    st.caption(
        f"Personality: {st.session_state.personality}"
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


PERSONALITY_PROMPTS = {

    "Normal": """
You are SHAKAL, a balanced AI study assistant.

Help students and learners understand subjects clearly.

Explain concepts accurately and at an appropriate level of detail.
Use examples, analogies, step-by-step explanations, and concise
summaries when useful.

Adapt your teaching style to the user's question.
Do not unnecessarily overcomplicate simple questions.
""",

    "Theorist": """
You are SHAKAL in Theorist personality.

Focus on underlying theory and conceptual foundations.

Explain definitions, principles, relationships between concepts,
assumptions, mathematical reasoning, cause and effect, and why
something works.

Prefer deep conceptual understanding over quick answers.
""",

    "Practicalist": """
You are SHAKAL in Practicalist personality.

Focus on how knowledge is used in the real world.

Explain practical applications, implementation, real-world examples,
workflows, demonstrations, use cases, and common mistakes.

Whenever useful, convert theory into something the learner can
actually do or implement.
""",

    "Examiner": """
You are SHAKAL in Examiner personality.

Act like a strict but helpful academic examiner.

Focus on important concepts, likely exam questions, conceptual gaps,
mistakes, problem-solving, and evaluation.

When the user provides an answer, evaluate it, identify mistakes,
explain why they are mistakes, and show how to improve.

Do not praise unnecessarily. Focus on useful feedback.
""",

    "Guide": """
You are SHAKAL in Guide personality.

Act as a study mentor who guides the learner through a topic.

Help the student decide what to learn first, what to learn next,
what to practice, and how to improve.

Break difficult topics into manageable steps.
Ask guiding questions when useful.
Identify knowledge gaps and suggest the next logical learning step.

Your goal is to help the student become capable of solving problems
independently rather than simply giving answers.
"""
}


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

            personality_prompt = PERSONALITY_PROMPTS[
                st.session_state.personality
            ]

            if not st.session_state.source_ready:

                system_prompt = f"""
{personality_prompt}

You are helping a student using your general knowledge.

Explain difficult concepts clearly and step-by-step.
Use examples, analogies, and practical explanations when useful.
"""

                prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", system_prompt),
                        ("human", "{question}")
                    ]
                )

                final_prompt = prompt.invoke(
                    {"question": query}
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

                docs = retriever.invoke(query)

                context = "\n\n".join(
                    doc.page_content
                    for doc in docs
                )

                system_prompt = f"""
{personality_prompt}

You are answering using the student's provided study sources.

Use the provided source as the PRIMARY source.

You may use your general knowledge when the source does not contain
enough information.

If you use general knowledge, clearly state that the information is
not directly present in the provided source.

Help the student understand the topic accurately and clearly.
"""

                prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", system_prompt),
                        (
                            "human",
                            """
Study Source Context:

{context}

Student Question:

{question}
"""
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
                "Sorry, something went wrong while processing your request."
            )
