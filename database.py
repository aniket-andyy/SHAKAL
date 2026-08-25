import base64
import io
import os

from PIL import Image
from langchain_core.documents import Document
from langchain_mistralai import ChatMistralAI

MIME_MAP = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}

def _image_to_base64(path, max_side=1568, quality=85):
    """Resize + re-encode so big screenshots never blow the API payload limit."""
    with Image.open(path) as img:
        img = img.convert("RGB")
        img.thumbnail((max_side, max_side))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        return base64.b64encode(buffer.getvalue()).decode()

def load_image(path, chunk_size=1800, chunk_overlap=200):
    mime = MIME_MAP.get(os.path.splitext(path)[1].lower(), "image/jpeg")
    b64 = _image_to_base64(path)

    vision_llm = ChatMistralAI(model="mistral-small-2506", max_retries=1)

    response = vision_llm.invoke([
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "This image is study material. First extract ALL visible "
                        "text word-for-word. Then describe every diagram, chart, "
                        "table, formula and concept in clear detailed language so "
                        "a student can study using only your output."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                },
            ],
        }
    ])

    text = (response.content or "").strip()

    if not text:
        return []   # NEVER store empty docs

    docs = []
    step = chunk_size - chunk_overlap
    for start in range(0, len(text), step):
        chunk = text[start:start + chunk_size].strip()
        if chunk:
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": os.path.basename(path),
                        "source_type": "image",
                    },
                )
            )
    return docs
