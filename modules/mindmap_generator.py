"""
modules/mindmap_generator.py
────────────────────────────
AI Mind Map & Concept Map Generator — backend module.

Responsibilities:
  • Build prompts for Ollama
  • Parse + validate the returned JSON hierarchy
  • Assign branch colours
  • Return visualization-ready dicts

Uses the existing Ollama pipeline only — no OpenAI, no HuggingFace.
"""

import json
import re
import requests
import logging

logger = logging.getLogger(__name__)

# ── Ollama config (same base URL your RAG pipeline uses) ──────────────────────
OLLAMA_URL  = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral:latest"         # change to whatever model you have pulled

# ── Colour palette for branches (cycles if tree is deep) ─────────────────────
BRANCH_COLOURS = [
    "#6366f1",   # indigo
    "#8b5cf6",   # violet
    "#06b6d4",   # cyan
    "#10b981",   # emerald
    "#f59e0b",   # amber
    "#ef4444",   # red
    "#ec4899",   # pink
    "#3b82f6",   # blue
    "#14b8a6",   # teal
    "#f97316",   # orange
]

# ─────────────────────────────────────────────────────────────────────────────
# PROMPT BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def _doc_prompt(text: str) -> str:
    """Prompt for document → mindmap."""
    snippet = text[:3000]   # keep prompt size manageable
    return f"""You are an expert knowledge cartographer.
Analyse the document excerpt below and extract its core topics and sub-concepts.
Return ONLY valid JSON — no markdown fences, no explanation.

JSON schema (unlimited depth):
{{
  "name": "<main topic>",
  "children": [
    {{
      "name": "<branch>",
      "children": [
        {{
          "name": "<sub-concept>",
          "children": []
        }}
      ]
    }}
  ]
}}

Rules:
- Root node must capture the document's central theme.
- 4–8 top-level branches covering the major themes.
- Each branch may have 2–5 children; children may have further children.
- Node names must be concise (3–7 words max).
- No duplicate names.
- Output ONLY the JSON object.

Document excerpt:
\"\"\"
{snippet}
\"\"\"
"""


def _chat_prompt(user_prompt: str) -> str:
    """Prompt for free-text chat → mindmap."""
    return f"""You are an expert knowledge cartographer.
The user wants a mind map for: "{user_prompt}"

Return ONLY valid JSON — no markdown fences, no explanation.

JSON schema (unlimited depth):
{{
  "name": "<central topic>",
  "children": [
    {{
      "name": "<branch>",
      "children": [
        {{
          "name": "<sub-concept>",
          "children": []
        }}
      ]
    }}
  ]
}}

Rules:
- Root captures the topic exactly.
- 5–8 meaningful top-level branches.
- Each branch: 2–5 children.
- Concise node names (3–7 words).
- No duplicate names.
- Output ONLY the JSON object.
"""


# ─────────────────────────────────────────────────────────────────────────────
# OLLAMA CALL
# ─────────────────────────────────────────────────────────────────────────────

def _call_ollama(prompt: str) -> str:
    """Send prompt to local Ollama and return the raw text response."""
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": 2048},
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()
    return response.json().get("response", "")


# ─────────────────────────────────────────────────────────────────────────────
# JSON PARSING & VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def _extract_json(raw: str) -> dict:
    """
    Robustly extract the first {...} JSON block from a raw LLM response.
    Strips markdown fences if present.
    """
    # Remove markdown code fences
    raw = re.sub(r"```(?:json)?", "", raw).strip()

    # Find the first { ... } block
    start = raw.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model response.")

    # Balance braces to find the closing }
    depth = 0
    end = start
    for i, ch in enumerate(raw[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        if depth == 0:
            end = i
            break

    json_str = raw[start : end + 1]
    return json.loads(json_str)


def _validate_node(node: dict, depth: int = 0) -> dict:
    """
    Recursively ensure every node has 'name' (str) and 'children' (list).
    Truncates at depth 5 to prevent runaway trees.
    """
    if not isinstance(node, dict):
        raise ValueError(f"Node is not a dict: {node}")
    if "name" not in node or not isinstance(node["name"], str):
        raise ValueError(f"Node missing valid 'name': {node}")

    node.setdefault("children", [])
    if not isinstance(node["children"], list):
        node["children"] = []

    if depth >= 5:
        node["children"] = []          # cap depth
        return node

    node["children"] = [
        _validate_node(child, depth + 1)
        for child in node["children"]
        if isinstance(child, dict)
    ]
    return node


# ─────────────────────────────────────────────────────────────────────────────
# COLORISER
# ─────────────────────────────────────────────────────────────────────────────

def _colorise(node: dict, colour_idx: int = 0, is_root: bool = True) -> dict:
    """
    Attach a 'color' key to every node.
    Root gets a special gradient-start colour; each top-level branch
    gets a distinct hue that its children inherit (slightly lighter).
    """
    if is_root:
        node["color"] = "#a78bfa"       # root: soft violet
        for i, child in enumerate(node.get("children", [])):
            _colorise(child, i, is_root=False)
    else:
        node["color"] = BRANCH_COLOURS[colour_idx % len(BRANCH_COLOURS)]
        child_colour = node["color"]
        for child in node.get("children", []):
            child["color"] = child_colour
            for grandchild in child.get("children", []):
                _colorise(grandchild, colour_idx, is_root=False)

    return node


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def generate_mindmap(text: str = "", prompt: str = "", source: str = "chat") -> dict:
    """
    Main entry point.

    Parameters
    ----------
    text   : str   — Raw document text (used when source="document")
    prompt : str   — User chat message  (used when source="chat")
    source : str   — "document" | "chat"

    Returns
    -------
    dict  — Validated, colourised mindmap hierarchy ready for D3.js rendering.
            On failure, returns a minimal error tree.
    """
    try:
        if source == "document":
            if not text.strip():
                raise ValueError("Document text is empty.")
            llm_prompt = _doc_prompt(text)
        else:
            if not prompt.strip():
                raise ValueError("Chat prompt is empty.")
            llm_prompt = _chat_prompt(prompt)

        raw = _call_ollama(llm_prompt)
        tree = _extract_json(raw)
        tree = _validate_node(tree)
        tree = _colorise(tree)
        return {"success": True, "data": tree}

    except requests.exceptions.ConnectionError:
        logger.error("Ollama not reachable.")
        return {
            "success": False,
            "error": "Cannot reach Ollama. Make sure it is running on localhost:11434.",
            "data": _fallback_tree("Ollama Offline"),
        }
    except (json.JSONDecodeError, ValueError) as exc:
        logger.error("JSON parse error: %s", exc)
        return {
            "success": False,
            "error": f"Could not parse model response: {exc}",
            "data": _fallback_tree("Parse Error"),
        }
    except Exception as exc:
        logger.error("Unexpected error: %s", exc)
        return {
            "success": False,
            "error": str(exc),
            "data": _fallback_tree("Generation Error"),
        }


def _fallback_tree(label: str) -> dict:
    """Minimal placeholder tree shown on error."""
    return {
        "name": label,
        "color": "#ef4444",
        "children": [
            {"name": "Check Ollama", "color": "#f97316", "children": []},
            {"name": "Retry Request", "color": "#f59e0b", "children": []},
        ],
    }

