from modules.llm import ask_llm


def analyze_paper(text):
    prompt = f"""
Analyze this research paper:

- Title
- Abstract
- Methodology
- Results
- Conclusion
- Keywords
- Strengths
- Limitations
- Future Work

Paper:
{text}
"""

    return ask_llm(prompt, mode="balanced")