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
import base64
import mimetypes


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
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    for doc in docs:
        doc.metadata["source_type"] = "pdf"
        doc.metadata["source"] = pdf_path

    return docs


def load_webpage(url):
    loader = WebBaseLoader(url)
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
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    raise ValueError("Invalid YouTube URL.")


def load_youtube(url):
    video_id = extract_youtube_video_id(url)

    try:
        # youtube_transcript_api >= 1.0
        api = YouTubeTranscriptApi()
        transcript = api.fetch(video_id)
        text = " ".join(
            segment.text if hasattr(segment, "text") else segment["text"]
            for segment in transcript
        )
    except Exception:
        # legacy API fallback
        raw = YouTubeTranscriptApi.get_transcript(video_id)
        text = " ".join(item["text"] for item in raw)

    if not text.strip():
        raise ValueError("YouTube transcript is empty.")

    document = Document(
        page_content=text,
        metadata={
            "source_type": "youtube",
            "source": url,
            "video_id": video_id
        }
    )

    return [document]


# ==========================================
# IMAGE EXTRACTION — DOUBLE FALLBACK SYSTEM
# ==========================================
def _read_image_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(
            image_file.read()
        ).decode("utf-8")

    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type:
        mime_type = "image/jpeg"

    return encoded, mime_type


def _extract_with_ocr(base64_string, mime_type, api_key):
    """Method 1: Mistral dedicated OCR endpoint (JSON + base64 data URL)."""
    payload = {
        "model": "mistral-ocr-latest",
        "document": {
            "type": "image_url",
            "image_url": f"data:{mime_type};base64,{base64_string}",
        },
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    response = requests.post(
        "https://api.mistral.ai/v1/ocr",
        headers=headers,
        json=payload,
        timeout=120,
    )

    response.raise_for_status()
    result = response.json()

    parts = [
        page.get("markdown", "")
        for page in result.get("pages", [])
        if page.get("markdown")
    ]

    return "\n\n".join(parts)


def _extract_with_vision_llm(base64_string, mime_type):
    """Method 2: Fallback — Mistral Small vision via chat completions."""
    from langchain_mistralai import ChatMistralAI

    llm = ChatMistralAI(model="mistral-small-2506")

    response = llm.invoke(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "This image is study material. Extract ALL visible "
                            "text word-for-word, then describe every diagram, "
                            "chart, table, formula and concept in clear detail "
                            "so a student can study using only your output."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{base64_string}"
                        },
                    },
                ],
            }
        ]
    )

    return response.content


def load_image(image_path):
    api_key = os.getenv("MISTRAL_API_KEY")

    if not api_key:
        raise ValueError("MISTRAL_API_KEY not found.")

    base64_string, mime_type = _read_image_base64(image_path)

    text = ""
    errors = []

    # Try Method 1: OCR endpoint
    try:
        text = _extract_with_ocr(base64_string, mime_type, api_key)
    except Exception as e:
        errors.append(f"OCR: {e}")

    # If OCR failed or returned nothing, try Method 2: Vision LLM
    if not text or not text.strip():
        try:
            text = _extract_with_vision_llm(base64_string, mime_type)
        except Exception as e:
            errors.append(f"Vision: {e}")

    if not text or not text.strip():
        raise ValueError(
            "No readable text found in image. " + " | ".join(errors)
        )

    document = Document(
        page_content=text.strip(),
        metadata={
            "source_type": "image",
            "source": os.path.basename(image_path),
        }
    )

    return [document]


def add_to_chroma(docs):
    if not docs:
        raise ValueError("No documents were provided.")

    chunks = splitter.split_documents(docs)

    # Safety: drop blank chunks (prevents Chroma crashes)
    chunks = [
        c for c in chunks
        if c.page_content and c.page_content.strip()
    ]

    if not chunks:
        raise ValueError("No valid text chunks were created.")

    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_model
    )

    vectorstore.add_documents(documents=chunks)

    return vectorstore


def clear_database():
    if os.path.exists(CHROMA_PATH):
        import shutil
        shutil.rmtree(CHROMA_PATH)


if __name__ == "__main__":

    print("\n======================================")
    print("        RAG DATABASE BUILDER")
    print("======================================")
    print("\nSelect source:")
    print("1. PDF")
    print("2. Webpage")
    print("3. YouTube URL")
    print("4. Image")

    choice = input("\nChoose source (1/2/3/4): ").strip()

    try:
        if choice == "1":
            path = input("Enter PDF path: ").strip()
            add_to_chroma(load_pdf(path))

        elif choice == "2":
            url = input("Enter webpage URL: ").strip()
            add_to_chroma(load_webpage(url))

        elif choice == "3":
            url = input("Enter YouTube URL: ").strip()
            add_to_chroma(load_youtube(url))

        elif choice == "4":
            path = input("Enter image path: ").strip()
            add_to_chroma(load_image(path))

        else:
            print("Invalid selection.")

        print("\n✅ Successfully added to database!")

    except Exception as e:
        print(f"\n❌ Error processing source: {e}")
