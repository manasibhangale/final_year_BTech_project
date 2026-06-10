"""
modules/flashcard_generator.py
───────────────────────────────
AI Flashcard Generator — backend module.

Responsibilities:
  • Build LLM prompts for flashcard generation
  • Parse + validate returned JSON
  • Assign difficulty colours / metadata
  • Return study-ready flashcard list

Uses the existing Ollama pipeline only.
"""

import json
import re
import requests
import logging
import time

logger = logging.getLogger(__name__)

# ── Ollama config ─────────────────────────────────────────────────────────────
OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral:latest"         # change to whichever model you have pulled

# ── Difficulty colour map ─────────────────────────────────────────────────────
DIFFICULTY_COLOURS = {
    "Easy":   {"bg": "rgba(16,185,129,0.15)",  "border": "rgba(16,185,129,0.35)",  "text": "#6ee7b7"},
    "Medium": {"bg": "rgba(245,158,11,0.15)",  "border": "rgba(245,158,11,0.35)",  "text": "#fbbf24"},
    "Hard":   {"bg": "rgba(239,68,68,0.15)",   "border": "rgba(239,68,68,0.35)",   "text": "#fca5a5"},
}

MEMORY_LABELS = {0: "New", 1: "Learning", 2: "Familiar", 3: "Mastered"}
MEMORY_COLOURS = {
    0: "#6366f1",   # new — indigo
    1: "#f59e0b",   # learning — amber
    2: "#06b6d4",   # familiar — cyan
    3: "#10b981",   # mastered — emerald
}


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def _build_prompt(text: str, num_cards: int = 15) -> str:
    snippet = text[:4000]
    return f"""You are an expert academic tutor and flashcard creator.
Analyse the text below and generate exactly {num_cards} high-quality flashcards.

Return ONLY a valid JSON array — no markdown fences, no explanation, nothing else.

Each flashcard must follow this exact schema:
{{
  "question": "<clear, concise question>",
  "answer": "<accurate, complete answer>",
  "topic": "<specific topic/chapter name>",
  "difficulty": "Easy" | "Medium" | "Hard",
  "hint": "<short hint to jog memory>",
  "explanation": "<1-2 sentence deeper explanation>"
}}

Card type distribution:
- 5 definition cards  (What is X?)
- 4 concept cards     (Explain how / Why does...)
- 3 application cards (What happens when... / How would you...)
- 3 formula/rule cards (What is the formula/rule for...)

Rules:
- Questions must be specific and unambiguous.
- Answers must be factual and taken from the document.
- Topics must reflect actual sections in the text.
- Difficulty must be one of: Easy, Medium, Hard.
- No duplicate questions.
- Output ONLY the JSON array, starting with [ and ending with ].

Text:
\"\"\"
{snippet}
\"\"\"
"""


# ─────────────────────────────────────────────────────────────────────────────
# OLLAMA CALL
# ─────────────────────────────────────────────────────────────────────────────

def _call_ollama(prompt: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.4, "num_predict": 3000},
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=180)
    resp.raise_for_status()
    return resp.json().get("response", "")


# ─────────────────────────────────────────────────────────────────────────────
# JSON PARSING
# ─────────────────────────────────────────────────────────────────────────────

def _extract_json_array(raw: str) -> list:
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    start = raw.find("[")
    if start == -1:
        raise ValueError("No JSON array found in model response.")
    depth, end = 0, start
    for i, ch in enumerate(raw[start:], start):
        if ch == "[":  depth += 1
        elif ch == "]": depth -= 1
        if depth == 0:
            end = i
            break
    return json.loads(raw[start:end + 1])


def _validate_card(card: dict, idx: int) -> dict:
    """Ensure required keys exist and are the right type."""
    defaults = {
        "question":    f"Question {idx + 1}",
        "answer":      "—",
        "topic":       "General",
        "difficulty":  "Medium",
        "hint":        "",
        "explanation": "",
    }
    for k, v in defaults.items():
        if k not in card or not card[k]:
            card[k] = v
    if card["difficulty"] not in ("Easy", "Medium", "Hard"):
        card["difficulty"] = "Medium"
    # Spaced-repetition fields
    card.setdefault("memory_level", 0)
    card.setdefault("correct_streak", 0)
    card.setdefault("times_seen", 0)
    card.setdefault("last_seen", None)
    card.setdefault("starred", False)
    # Colour metadata
    card["diff_style"] = DIFFICULTY_COLOURS.get(card["difficulty"], DIFFICULTY_COLOURS["Medium"])
    card["mem_colour"] = MEMORY_COLOURS.get(card["memory_level"], "#6366f1")
    card["mem_label"]  = MEMORY_LABELS.get(card["memory_level"], "New")
    return card


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def generate_flashcards(text: str, num_cards: int = 15) -> dict:
    """
    Generate flashcards from raw document text.

    Returns
    -------
    dict with keys:
        success  : bool
        cards    : list[dict]
        error    : str  (only on failure)
    """
    try:
        if not text.strip():
            raise ValueError("Document text is empty.")

        prompt = _build_prompt(text, num_cards)
        raw    = _call_ollama(prompt)
        cards  = _extract_json_array(raw)

        if not cards:
            raise ValueError("Model returned an empty array.")

        validated = [_validate_card(c, i) for i, c in enumerate(cards)]
        return {"success": True, "cards": validated}

    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "error": "Cannot reach Ollama. Make sure it is running on localhost:11434.",
            "cards": [],
        }
    except (json.JSONDecodeError, ValueError) as exc:
        logger.error("Flashcard parse error: %s", exc)
        return {"success": False, "error": f"Could not parse model response: {exc}", "cards": []}
    except Exception as exc:
        logger.error("Unexpected flashcard error: %s", exc)
        return {"success": False, "error": str(exc), "cards": []}


def update_memory(card: dict, knew_it: bool) -> dict:
    """
    Update a card's spaced-repetition fields after a study response.

    knew_it=True  → memory_level up (max 3), streak++
    knew_it=False → memory_level reset to max(0, level-1), streak=0
    """
    card["times_seen"] = card.get("times_seen", 0) + 1
    card["last_seen"]  = time.time()

    if knew_it:
        card["correct_streak"] = card.get("correct_streak", 0) + 1
        if card["correct_streak"] >= 2:
            card["memory_level"] = min(3, card.get("memory_level", 0) + 1)
    else:
        card["correct_streak"] = 0
        card["memory_level"]   = max(0, card.get("memory_level", 0) - 1)

    card["mem_colour"] = MEMORY_COLOURS.get(card["memory_level"], "#6366f1")
    card["mem_label"]  = MEMORY_LABELS.get(card["memory_level"], "New")
    return card


def get_analytics(cards: list) -> dict:
    """Compute analytics summary from the current card deck."""
    total     = len(cards)
    if total == 0:
        return {}

    mastered  = sum(1 for c in cards if c.get("memory_level", 0) == 3)
    learning  = sum(1 for c in cards if c.get("memory_level", 0) == 1)
    new_cards = sum(1 for c in cards if c.get("memory_level", 0) == 0)
    familiar  = sum(1 for c in cards if c.get("memory_level", 0) == 2)
    starred   = sum(1 for c in cards if c.get("starred", False))
    seen      = sum(1 for c in cards if c.get("times_seen", 0) > 0)

    # Per-topic breakdown
    topic_map: dict = {}
    for c in cards:
        t = c.get("topic", "General")
        topic_map.setdefault(t, {"total": 0, "mastered": 0})
        topic_map[t]["total"] += 1
        if c.get("memory_level", 0) == 3:
            topic_map[t]["mastered"] += 1

    # Difficulty split
    diff_map = {"Easy": 0, "Medium": 0, "Hard": 0}
    for c in cards:
        diff_map[c.get("difficulty", "Medium")] += 1

    mastery_pct = int(mastered / total * 100) if total else 0

    # Weak topics = topics with < 50 % mastery and > 1 card
    weak_topics = [
        t for t, v in topic_map.items()
        if v["total"] > 1 and (v["mastered"] / v["total"]) < 0.5
    ]

    return {
        "total": total, "mastered": mastered, "learning": learning,
        "familiar": familiar, "new": new_cards, "starred": starred,
        "seen": seen, "mastery_pct": mastery_pct,
        "topic_map": topic_map, "diff_map": diff_map,
        "weak_topics": weak_topics,
    }
