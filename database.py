from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from youtube_transcript_api import YouTubeTranscriptApi

import requests
import re
import os

load_dotenv()

CHROMA_PATH = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)


def load_pdf(pdf_path):
    print(f"\nLoading PDF: {pdf_path}")

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    print(f"Loaded {len(docs)} PDF pages.")

    return docs


def load_webpage(url):
    print(f"\nLoading webpage: {url}")

    loader = WebBaseLoader(url)
    docs = loader.load()

    print(f"Loaded {len(docs)} webpage documents.")

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
    print(f"\nLoading YouTube transcript: {url}")

    video_id = extract_youtube_video_id(url)

    api = YouTubeTranscriptApi()
    transcript = api.fetch(video_id)

    text = " ".join(
        segment.text
        for segment in transcript
    )

    document = Document(
        page_content=text,
        metadata={
            "source_type": "youtube",
            "source": url,
            "video_id": video_id
        }
    )

    print("YouTube transcript extracted.")

    return [document]


def load_image(image_path):
    print(f"\nProcessing image: {image_path}")

    api_key = os.getenv("MISTRAL_API_KEY")

    if not api_key:
        raise ValueError(
            "MISTRAL_API_KEY not found in .env"
        )

    with open(image_path, "rb") as image_file:
        response = requests.post(
            "https://api.mistral.ai/v1/ocr",
            headers={
                "Authorization": f"Bearer {api_key}"
            },
            files={
                "file": image_file
            },
            data={
                "model": "mistral-ocr-latest"
            }
        )

    response.raise_for_status()

    result = response.json()

    text_parts = []

    for page in result.get("pages", []):
        page_markdown = page.get("markdown", "")

        if page_markdown:
            text_parts.append(page_markdown)

    text = "\n\n".join(text_parts)

    if not text:
        raise ValueError(
            "No text could be extracted from the image."
        )

    document = Document(
        page_content=text,
        metadata={
            "source_type": "image",
            "source": image_path
        }
    )

    print("Image OCR completed.")

    return [document]


def add_to_chroma(docs):
    if not docs:
        print("No documents found.")
        return

    chunks = splitter.split_documents(docs)

    print(f"Created {len(chunks)} chunks.")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=CHROMA_PATH
    )

    print("Documents successfully stored in Chroma.")


print("\n======================================")
print("        RAG DATABASE BUILDER")
print("======================================")

print("\nSelect source:")
print("1. PDF")
print("2. Webpage")
print("3. YouTube URL")
print("4. Image")

choice = input(
    "\nChoose source (1/2/3/4): "
).strip()

if choice == "1":
    path = input(
        "Enter PDF path: "
    ).strip()

    docs = load_pdf(path)
    add_to_chroma(docs)

elif choice == "2":
    url = input(
        "Enter webpage URL: "
    ).strip()

    docs = load_webpage(url)
    add_to_chroma(docs)

elif choice == "3":
    url = input(
        "Enter YouTube URL: "
    ).strip()

    docs = load_youtube(url)
    add_to_chroma(docs)

elif choice == "4":
    path = input(
        "Enter image path: "
    ).strip()

    docs = load_image(path)
    add_to_chroma(docs)

else:
    print("\nInvalid selection.")

print("\nDatabase operation completed.") 
