import ollama
from utils.config import MODEL_FAST, MODEL_BALANCED


def ask_llm(prompt, mode="balanced"):
    model = MODEL_BALANCED if mode == "balanced" else MODEL_FAST

    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]