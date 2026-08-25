from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

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

llm = ChatMistralAI(
    model="mistral-small-2506"
)

source_llm_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a Study AI Assistant designed to help students learn,
understand, revise, and explore academic topics.

Your goal is to make studying easier and clearer.

The provided context comes from the user's study source.

Use the provided source as the PRIMARY source for answering the question.

You may use your general knowledge when the source does not contain
enough information to answer the question.

When explaining a difficult concept, explain it clearly and step-by-step.
Use examples, analogies, definitions, and practical explanations when
they help the student understand the topic.

If you use general knowledge, clearly state that the information is not
directly present in the provided source.

Do not contradict the source without explaining the difference."""
        ),
        (
            "human",
            """Study Source Context:

{context}

Student's Question:

{question}"""
        )
    ]
)

source_only_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a Study AI Assistant designed to help students
understand and learn from their provided study material.

You MUST answer the question using ONLY the provided source.

Do NOT use your general knowledge.

Do NOT invent, assume, or add information that is not supported by
the provided source.

Explain the answer clearly and in a student-friendly way while staying
strictly within the information contained in the source.

If the answer is not present in the provided source, say exactly:

"I could not find the answer in the provided source." """
        ),
        (
            "human",
            """Study Source Context:

{context}

Student's Question:

{question}"""
        )
    ]
)

llm_only_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a Study AI Assistant designed to help students
learn, understand, revise, and explore academic topics.

Your goal is to make studying easier and more effective.

Answer the student's question using your general knowledge.

Explain difficult concepts clearly and step-by-step.

When useful, provide:
- Simple explanations
- Examples
- Analogies
- Key points
- Practical applications
- Comparisons
- Short summaries

Adapt the depth of the explanation to the student's question.

There is no external study source available."""
        ),
        (
            "human",
            """Student's Question:

{question}"""
        )
    ]
)

print("\n==========================================")
print("          STUDY AI ASSISTANT")
print("==========================================")

print("\nSelect working mode:")
print("1. Source + LLM  [DEFAULT]")
print("2. Source Only")
print("3. LLM Only")

mode = input("\nChoose mode (1/2/3): ").strip()

if mode not in ["1", "2", "3"]:
    mode = "1"

if mode == "1":
    print("\nMode selected: Source + LLM")
elif mode == "2":
    print("\nMode selected: Source Only")
else:
    print("\nMode selected: LLM Only")

print("\nType 0 to exit.")
print("------------------------------------------\n")

while True:

    query = input("You : ").strip()

    if query == "0":
        print("\nExiting...")
        break

    if not query:
        continue

    if mode == "3":

        final_prompt = llm_only_prompt.invoke({
            "question": query
        })

        response = llm.invoke(final_prompt)

        print(f"\nAI: {response.content}\n")

        continue

    docs = retriever.invoke(query)

    if not docs:

        print(
            "\nAI: I could not find the answer "
            "in the provided source.\n"
        )

        continue

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    if mode == "1":

        final_prompt = source_llm_prompt.invoke({
            "context": context,
            "question": query
        })

    elif mode == "2":

        final_prompt = source_only_prompt.invoke({
            "context": context,
            "question": query
        })

    response = llm.invoke(final_prompt)

    print(f"\nAI: {response.content}\n") 
