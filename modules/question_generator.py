from modules.llm import ask_llm


def generate_questions(text, difficulty="Medium"):
    prompt = f"""
Generate a question paper from the content.

Difficulty: {difficulty}

Format:
Section A: MCQs
Section B: Short Answers
Section C: Long Answers

Content:
{text}
"""

    return ask_llm(prompt, mode="balanced")