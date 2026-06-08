"""
modules/quiz_generator.py
─────────────────────────
Generates an educational MCQ + True/False quiz from indexed document text.
Works by calling the project's existing generate_answer() pipeline with a
structured prompt, then parsing the JSON response.

ONLY MCQ and True/False — no short-answer — as requested.
"""

import json
import re
import warnings
import logging

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)


# ── Prompt template ───────────────────────────────────────────────────────────
# We ask for JSON only, instruct the model firmly, and chunk the text to stay
# within token limits.

MCQ_PROMPT = """You are a strict academic quiz generator. Based ONLY on the document text provided, create a quiz.

RULES:
- Every question MUST be about specific facts, concepts, or information actually present in the text.
- Do NOT create generic or abstract questions. Use real details from the text.
- For MCQ: provide exactly 4 options labeled A, B, C, D. Only one is correct.
- For True/False: write a factual statement from the text; answer is "True" or "False".
- Assign each question a short topic label (2-4 words) based on the text section it comes from.
- Return ONLY a valid JSON array. No markdown. No preamble. No explanation outside JSON.

Create EXACTLY:
- 7 MCQ questions
- 5 True/False questions

JSON format (strict):
[
  {{
    "type": "mcq",
    "topic": "<topic from document>",
    "question": "<specific question about document content>",
    "options": ["A) <option>", "B) <option>", "C) <option>", "D) <option>"],
    "answer": "<letter only: A or B or C or D>",
    "explanation": "<one sentence citing the document>"
  }},
  {{
    "type": "truefalse",
    "topic": "<topic from document>",
    "question": "<factual statement from document>",
    "answer": "<True or False>",
    "explanation": "<one sentence citing the document>"
  }}
]

DOCUMENT TEXT:
\"\"\"
{text}
\"\"\"

Return ONLY the JSON array starting with [ and ending with ]. Nothing else."""


# ── Main generate function ────────────────────────────────────────────────────

def generate_quiz(text: str):
    """
    Generate quiz from document text using the project's own generate_answer pipeline.
    Returns list[dict] of question objects.
    """
    # Use first 3500 chars to stay within LLM context limits
    text_snippet = text[:3500].strip()

    prompt = MCQ_PROMPT.format(text=text_snippet)

    # Use the project's RAG pipeline (generate_answer needs index + texts)
    try:
        from modules.rag_pipeline import generate_answer
        from modules.vector_store import load_index

        index, texts = load_index()
        if index is None:
            return []

        # We pass the full prompt as the "query" and let generate_answer use
        # the indexed context. The prompt already contains the text snippet so
        # the LLM always has content to work from.
        raw_answer, _ = generate_answer(prompt, index, texts)
        questions = _parse_json(raw_answer)
        if questions and len(questions) >= 3:
            return questions
    except Exception as e:
        logging.warning(f"RAG pipeline quiz generation failed: {e}")

    # Fallback: try a bare HuggingFace pipeline if available
    try:
        from transformers import pipeline as hf_pipeline
        gen = hf_pipeline(
            "text2text-generation",
            model="google/flan-t5-large",
            max_new_tokens=1200,
        )
        short_prompt = (
            "Generate 5 MCQ questions and 3 True/False questions from this text "
            "as a JSON array with fields type, topic, question, options (for MCQ), "
            "answer, explanation:\n\n" + text_snippet[:2000]
        )
        raw = gen(short_prompt)[0]["generated_text"]
        questions = _parse_json(raw)
        if questions and len(questions) >= 3:
            return questions
    except Exception as e:
        logging.warning(f"HF fallback quiz generation failed: {e}")

    return []


# ── JSON parser ───────────────────────────────────────────────────────────────

def _parse_json(raw: str) -> list:
    """Extract and validate a JSON array of question dicts from LLM output."""
    if not raw or not isinstance(raw, str):
        return []

    # Remove markdown fences
    raw = re.sub(r"```(?:json)?", "", raw, flags=re.IGNORECASE).strip()
    raw = raw.strip("`").strip()

    # Find outermost [ ... ]
    start = raw.find("[")
    end   = raw.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []

    try:
        data = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        # Try to salvage partial JSON by truncating at last complete object
        fragment = raw[start : end + 1]
        # Find last complete '}' before ']'
        last_brace = fragment.rfind("}")
        if last_brace > 0:
            try:
                data = json.loads(fragment[: last_brace + 1] + "]")
            except Exception:
                return []
        else:
            return []

    validated = []
    for item in data:
        if not isinstance(item, dict):
            continue
        q_type = item.get("type", "").lower().strip()
        if q_type not in ("mcq", "truefalse"):
            continue

        question = item.get("question", "").strip()
        answer   = item.get("answer", "").strip()
        topic    = item.get("topic", "General").strip()
        expl     = item.get("explanation", "").strip()

        if not question or not answer:
            continue

        if q_type == "mcq":
            options = item.get("options", [])
            if not isinstance(options, list) or len(options) < 2:
                continue
            # Normalise answer to single uppercase letter
            letter = re.match(r"([A-Da-d])", answer)
            if not letter:
                continue
            validated.append({
                "type":        "mcq",
                "topic":       topic,
                "question":    question,
                "options":     [str(o).strip() for o in options],
                "answer":      letter.group(1).upper(),
                "explanation": expl,
            })
        else:  # truefalse
            tf = answer.strip().lower()
            if "true" in tf:
                norm_ans = "True"
            elif "false" in tf:
                norm_ans = "False"
            else:
                continue
            validated.append({
                "type":        "truefalse",
                "topic":       topic,
                "question":    question,
                "answer":      norm_ans,
                "explanation": expl,
            })

    return validated


# ── Score + analytics ─────────────────────────────────────────────────────────

def analyze_quiz_results(questions: list, user_answers: dict) -> dict:
    """
    Score quiz and return analytics.

    user_answers: {question_index (int): answer_string}

    Returns:
        total, correct, score_pct,
        topic_scores {topic: {correct, total, pct}},
        strongest_topics, weakest_topics,
        recommendations,
        per_question [{question, correct_ans, user_ans, is_correct, topic, type}]
    """
    total   = len(questions)
    correct = 0
    topic_scores: dict[str, dict] = {}
    per_question = []

    for i, q in enumerate(questions):
        topic = q.get("topic", "General")
        if topic not in topic_scores:
            topic_scores[topic] = {"correct": 0, "total": 0}
        topic_scores[topic]["total"] += 1

        user_ans    = str(user_answers.get(i, "")).strip()
        correct_ans = str(q.get("answer", "")).strip()
        q_type      = q.get("type", "mcq")

        if q_type == "mcq":
            # Accept "A" or "A) ..." or "a"
            m = re.match(r"([A-Da-d])", user_ans)
            ua = m.group(1).upper() if m else ""
            is_correct = ua == correct_ans.upper()
        else:  # truefalse
            is_correct = user_ans.lower() == correct_ans.lower()

        if is_correct:
            correct += 1
            topic_scores[topic]["correct"] += 1

        per_question.append({
            "question":    q.get("question", ""),
            "correct_ans": correct_ans,
            "user_ans":    user_ans,
            "is_correct":  is_correct,
            "topic":       topic,
            "type":        q_type,
            "options":     q.get("options", []),
            "explanation": q.get("explanation", ""),
        })

    score_pct = round(correct / total * 100) if total else 0

    # Per-topic percentages
    for ts in topic_scores.values():
        ts["pct"] = round(ts["correct"] / ts["total"] * 100) if ts["total"] else 0

    sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1]["pct"], reverse=True)
    strongest = [t for t, s in sorted_topics if s["pct"] >= 70]
    weakest   = [t for t, s in sorted_topics if s["pct"] <  50]

    # Personalised recommendations
    recs = []
    for t in weakest[:3]:
        pct = topic_scores[t]["pct"]
        recs.append(f"Re-study **{t}** — you scored only {pct}% on this topic.")
    if score_pct >= 90:
        recs.append("🌟 Excellent! You have a strong grasp of the material.")
    elif score_pct >= 75:
        recs.append("🎯 Great job! Brush up on the weak topics to reach mastery.")
    elif score_pct >= 50:
        recs.append("📚 Decent effort. Review the highlighted weak topics carefully.")
    else:
        recs.append("🔄 Consider re-reading the document thoroughly before retaking.")

    return {
        "total":            total,
        "correct":          correct,
        "score_pct":        score_pct,
        "topic_scores":     topic_scores,
        "strongest_topics": strongest,
        "weakest_topics":   weakest,
        "recommendations":  recs,
        "per_question":     per_question,
    }
