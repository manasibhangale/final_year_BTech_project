import streamlit as st

def calculate_score(questions, user_answers):
    score = 0
    results = []

    for i, q in enumerate(questions):
        user_ans = user_answers.get(i)
        correct = q["answer"]

        is_correct = False

        if q["type"] == "mcq":
            is_correct = user_ans == correct
        else:
            is_correct = str(user_ans).lower() == str(correct).lower()

        if is_correct:
            score += 1

        results.append({
            "question": q["question"],
            "correct": correct,
            "user": user_ans,
            "explanation": q.get("explanation", ""),
            "is_correct": is_correct,
            "topic": q.get("topic", "General")
        })

    return score, results


def detect_weak_topics(results):
    topic_stats = {}

    for r in results:
        t = r["topic"]
        if t not in topic_stats:
            topic_stats[t] = {"correct": 0, "total": 0}

        topic_stats[t]["total"] += 1
        if r["is_correct"]:
            topic_stats[t]["correct"] += 1

    weak = []

    for t, v in topic_stats.items():
        acc = v["correct"] / v["total"]
        if acc < 0.5:
            weak.append(t)

    return weak


def adaptive_difficulty(score, total):
    pct = score / total

    if pct > 0.8:
        return "hard"
    elif pct > 0.5:
        return "medium"
    else:
        return "easy"