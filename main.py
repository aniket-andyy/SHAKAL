import os
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

personalities = {
    "Normal": """You are SHAKAL, a balanced AI study assistant.

Help students and learners understand subjects clearly.

Explain concepts accurately and at an appropriate level of detail.
Use examples, analogies, step-by-step explanations, and concise
summaries when useful.

Adapt your teaching style to the user's question.
Do not unnecessarily overcomplicate simple questions.""",

    "Theorist": """You are SHAKAL in Theorist personality.

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

When appropriate, connect the current concept with related theories
and fundamentals.""",

    "Practicalist": """You are SHAKAL in Practicalist personality.

Focus on how knowledge is used in the real world.

Explain:
- practical applications
- implementation
- real-world examples
- workflows
- demonstrations
- use cases
- common mistakes

Whenever useful, convert theory into something the learner can
actually do or implement.""",

    "Examiner": """You are SHAKAL in Examiner personality.

Act like a strict but helpful academic examiner.

Focus on:
- important concepts
- likely exam questions
- conceptual gaps
- mistakes
- problem-solving
- evaluation of answers

When the user provides an answer, evaluate it, identify mistakes,
explain why they are mistakes, and show how to improve.

Do not praise unnecessarily. Focus on useful feedback.""",

    "Guide": """You are SHAKAL in Guide personality.

Your primary role is to guide the student through learning.

Help the student understand what to learn, what to focus on, and
how to approach a problem.

Break difficult topics into manageable steps.

When solving a problem:
- identify what is being asked
- explain the approach
- guide the student through the reasoning
- provide hints when useful
- gradually move toward the solution

Help students build independent problem-solving skills instead of
simply giving answers.

Be patient, clear, supportive, and structured."""
}


def create_prompt(personality, mode):
    personality_instruction = personalities[personality]

    if mode == "1":
        return ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"""{personality_instruction}

You are also a Study AI Assistant designed to help students learn,
understand, revise, and explore academic topics.

The provided context comes from the student's study source.

Use the provided source as the PRIMARY source for answering the
question.

You may use your general knowledge when the source does not contain
enough information.

If you use general knowledge, clearly state that the information is
not directly present in the provided source.

Do not contradict the source without explaining the difference.

Always prioritize accuracy and educational value."""
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

    elif mode == "2":
        return ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"""{personality_instruction}

You are also a Study AI Assistant designed to help students learn
from their provided study material.

You MUST answer using ONLY the provided source.

Do NOT use your general knowledge.

Do NOT invent, assume, or add information that is not supported by
the provided source.

You may explain and organize the information differently to help
the student understand it, but every factual claim must be supported
by the source.

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

    else:
        return ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"""{personality_instruction}

You are also a Study AI Assistant designed to help students learn,
understand, revise, and explore academic topics.

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


# ==========================================
# FIX 1: Knowledge base health check
# ==========================================
def kb_count():
    """How many chunks are actually stored? Lets you verify image ingestion worked."""
    try:
        return vectorstore._collection.count()
    except Exception:
        try:
            return len(vectorstore.get()["ids"])
        except Exception:
            return 0


# ==========================================
# FIX 2: Safe retrieval (MMR crash guard + empty chunk filter)
# ==========================================
def safe_retrieve(query):
    docs = []
    try:
        docs = retriever.invoke(query)
    except Exception:
        # MMR can crash on tiny/empty collections -> fall back to similarity
        try:
            docs = vectorstore.similarity_search(query, k=4)
        except Exception:
            docs = []

    # FIX 3: drop blank chunks (classic failed-image-extraction residue)
    return [d for d in docs if (d.page_content or "").strip()]


print("\n==========================================")
print("             SHAKAL - STUDY MADAD")
print("==========================================")

print(f"\nKnowledge base: {kb_count()} chunks loaded")

print("\nSelect working mode:")
print("1. Source + LLM  [DEFAULT]")
print("2. Only My Source")
print("3. Only LLM")

mode = input("\nChoose mode (1/2/3): ").strip()

if mode not in ["1", "2", "3"]:
    mode = "1"

print("\nSelect AI personality:")

personality_names = list(personalities.keys())

for index, name in enumerate(personality_names, 1):
    print(f"{index}. {name}")

personality_choice = input(
    "\nChoose personality (1-5): "
).strip()

try:
    personality_index = int(personality_choice) - 1

    if personality_index not in range(len(personality_names)):
        personality_index = 0

except ValueError:
    personality_index = 0

personality = personality_names[personality_index]

prompt = create_prompt(
    personality,
    mode
)

print(f"\nWorking mode: {mode}")
print(f"Personality: {personality}")

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

        final_prompt = prompt.invoke({
            "question": query
        })

        response = llm.invoke(final_prompt)

        print(f"\nAI: {response.content}\n")

        continue

    # NEW: friendly message instead of crash when DB is empty
    if kb_count() == 0:
        print(
            "\nAI: Your knowledge base is empty. "
            "Process your sources first (make sure the image "
            "contains readable text/diagrams).\n"
        )
        continue

    docs = safe_retrieve(query)

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

    if not context.strip():

        print(
            "\nAI: I could not find the answer "
            "in the provided source.\n"
        )

        continue

    final_prompt = prompt.invoke(
        {
            "context": context,
            "question": query
        }
    )

    response = llm.invoke(final_prompt)

    print(f"\nAI: {response.content}\n")
