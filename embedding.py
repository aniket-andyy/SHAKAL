from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

texts = [
    "Hello, this is Aniket.",
    "Hello, your name is YouTube.",
    "And you all are very good."
]

vectors = embedding.embed_documents(texts)

print(f"Number of texts: {len(texts)}")
print(f"Embedding dimensions: {len(vectors[0])}")

for i, vector in enumerate(vectors):
    print(f"\nText {i + 1}: {texts[i]}")
    print(f"Vector: {vector}")
