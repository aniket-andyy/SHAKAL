from dotenv import load_dotenv

from langchain_community.document_loaders import (
    PyPDFLoader,
    WebBaseLoader
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_huggingface import (
    HuggingFaceEmbeddings
)

from langchain_community.vectorstores import (
    Chroma
)

from langchain_core.documents import (
    Document
)

from youtube_transcript_api import (
    YouTubeTranscriptApi
)

import requests
import re
import os


load_dotenv()


CHROMA_PATH = "chroma_db"

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)


splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)


def load_pdf(pdf_path):

    loader = PyPDFLoader(
        pdf_path
    )

    docs = loader.load()

    for doc in docs:

        doc.metadata["source_type"] = "pdf"
        doc.metadata["source"] = pdf_path

    return docs


def load_webpage(url):

    loader = WebBaseLoader(
        url
    )

    docs = loader.load()

    for doc in docs:

        doc.metadata["source_type"] = "webpage"
        doc.metadata["source"] = url

    return docs


def extract_youtube_video_id(url):

    patterns = [
        r"(?:youtube\.com/watch\?v=)([^&]+)",
        r"(?:youtu\.be/)([^?]+)",
        r"(?:youtube\.com/shorts/)([^?]+)"
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            url
        )

        if match:

            return match.group(1)

    raise ValueError(
        "Invalid YouTube URL."
    )


def load_youtube(url):

    video_id = extract_youtube_video_id(
        url
    )

    api = YouTubeTranscriptApi()

    transcript = api.fetch(
        video_id
    )

    text = " ".join(
        segment.text
        for segment in transcript
    )

    if not text.strip():

        raise ValueError(
            "YouTube transcript is empty."
        )

    document = Document(
        page_content=text,
        metadata={
            "source_type": "youtube",
            "source": url,
            "video_id": video_id
        }
    )

    return [document]


def load_image(image_path):

    api_key = os.getenv(
        "MISTRAL_API_KEY"
    )

    if not api_key:

        raise ValueError(
            "MISTRAL_API_KEY not found."
        )

    with open(
        image_path,
        "rb"
    ) as image_file:

        response = requests.post(
            "https://api.mistral.ai/v1/ocr",
            headers={
                "Authorization":
                    f"Bearer {api_key}"
            },
            files={
                "file": image_file
            },
            data={
                "model":
                    "mistral-ocr-latest"
            },
            timeout=120
        )

    response.raise_for_status()

    result = response.json()

    text_parts = []

    for page in result.get(
        "pages",
        []
    ):

        markdown = page.get(
            "markdown",
            ""
        )

        if markdown:

            text_parts.append(
                markdown
            )

    text = "\n\n".join(
        text_parts
    )

    if not text.strip():

        raise ValueError(
            "No readable text found in image."
        )

    document = Document(
        page_content=text,
        metadata={
            "source_type": "image",
            "source": image_path
        }
    )

    return [document]


def add_to_chroma(docs):

    if not docs:

        raise ValueError(
            "No documents were provided."
        )

    chunks = splitter.split_documents(
        docs
    )

    if not chunks:

        raise ValueError(
            "No text chunks were created."
        )

    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_model
    )

    vectorstore.add_documents(
        documents=chunks
    )

    return vectorstore


def clear_database():

    if os.path.exists(
        CHROMA_PATH
    ):

        import shutil

        shutil.rmtree(
            CHROMA_PATH
        )


if __name__ == "__main__":

    print(
        "\n======================================"
    )

    print(
        "        RAG DATABASE BUILDER"
    )

    print(
        "======================================"
    )

    print(
        "\nSelect source:"
    )

    print(
        "1. PDF"
    )

    print(
        "2. Webpage"
    )

    print(
        "3. YouTube URL"
    )

    print(
        "4. Image"
    )

    choice = input(
        "\nChoose source (1/2/3/4): "
    ).strip()

    if choice == "1":

        path = input(
            "Enter PDF path: "
        ).strip()

        docs = load_pdf(
            path
        )

        add_to_chroma(
            docs
        )

    elif choice == "2":

        url = input(
            "Enter webpage URL: "
        ).strip()

        docs = load_webpage(
            url
        )

        add_to_chroma(
            docs
        )

    elif choice == "3":

        url = input(
            "Enter YouTube URL: "
        ).strip()

        docs = load_youtube(
            url
        )

        add_to_chroma(
            docs
        )

    elif choice == "4":

        path = input(
            "Enter image path: "
        ).strip()

        docs = load_image(
            path
        )

        add_to_chroma(
            docs
        )

    else:

        print(
            "Invalid selection."
    )
