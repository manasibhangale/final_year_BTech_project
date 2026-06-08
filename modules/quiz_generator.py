import json
import random
from collections import defaultdict
from modules.rag_pipeline import generate_answer


# ─────────────────────────────────────────────
# QUIZ GENERATION
# ─────────────────────────────────────────────
def generate_quiz(text, difficulty="Medium"):
    """
    Generates structured quiz in JSON format
    """

    prompt = f"""
You are an AI quiz generator.

Create a quiz from the following text.

Rules:
- Difficulty: {difficulty}
- 7 MCQ questions
- 5 True/False questions
- Each question must include:
  - question
  - options (for MCQ only, 4 options A-D)
  - correct answer
  - topic
  - explanation

Return ONLY valid JSON in this format:

{{
  "questions": [
    {{
      "type": "mcq",
      "topic": "some topic",
      "question": "...",
      "options": ["A ...", "B ...", "C ...", "D ..."],
      "answer": "A",
      "explanation": "..."
    }},
    {{
      "type": "truefalse",
      "topic": "some topic",
      "question": "...",
      "answer": "True",
      "explanation": "..."
    }}
  ]
}}

TEXT:
{text}
"""

    response = generate_answer(prompt, None, None)[0]

    try:
        data = json.loads(response)
        return data["questions"]
    except:
        return []


# ─────────────────────────────────────────────
# QUIZ ANALYSIS (UPGRADED)
# ─────────────────────────────────────────────
def analyze_quiz_results(questions, user_answers):
    """
    Returns detailed performance analytics
    """

    correct = 0
    topic_scores = defaultdict(lambda: {"correct": 0, "total": 0})

    per_question = []

    for i, q in enumerate(questions):
        q_type = q["type"]
        topic = q.get("topic", "General")
        correct_ans = q["answer"]

        user_ans = user_answers.get(i, "")

        is_correct = False

        if q_type == "mcq":
            is_correct = user_ans.upper() == correct_ans.upper()
        else:
            is_correct = str(user_ans).lower() == str(correct_ans).lower()

        topic_scores[topic]["total"] += 1
        if is_correct:
            correct += 1
            topic_scores[topic]["correct"] += 1

        per_question.append({
            "question": q["question"],
            "type": q_type,
            "topic": topic,
            "options": q.get("options", []),
            "correct_ans": correct_ans,
            "user_ans": user_ans,
            "is_correct": is_correct,
            "explanation": q.get("explanation", "")
        })

    total = len(questions)
    score_pct = int((correct / total) * 100) if total else 0

    # ── Weak / Strong Topics ──
    strongest = []
    weakest = []

    for topic, val in topic_scores.items():
        pct = (val["correct"] / val["total"]) * 100

        if pct >= 75:
            strongest.append(topic)
        elif pct <= 50:
            weakest.append(topic)

        topic_scores[topic]["pct"] = int(pct)

    # ── Recommendations ──
    recommendations = []

    for topic in weakest:
        recommendations.append(
            f"Revise '{topic}' — focus on core definitions and examples."
        )

    if score_pct < 50:
        recommendations.append("Re-read the document and retake quiz after revision.")
    elif score_pct < 75:
        recommendations.append("Practice more MCQs on weak topics.")
    else:
        recommendations.append("Try increasing difficulty to HARD mode.")

    return {
        "score_pct": score_pct,
        "correct": correct,
        "total": total,
        "topic_scores": dict(topic_scores),
        "strongest_topics": strongest,
        "weakest_topics": weakest,
        "recommendations": recommendations,
        "per_question": per_question
    }