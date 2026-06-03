from modules.vector_store import search
from modules.llm import ask_llm
from utils.config import TOP_K


def generate_answer(query, index, texts):
    chunks = search(query, index, texts, TOP_K)
    context = "\n".join(chunks)

    prompt = f"""
Answer ONLY from the context below.
If answer is not found, say "Not found in document".

Context:
{context}

Question:
{query}
"""

    answer = ask_llm(prompt, mode="balanced")

    return answer, chunks