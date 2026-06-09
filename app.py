import streamlit as st
import os
import time
import json
import streamlit.components.v1 as components

from modules.document_loader import load_document, load_url
from modules.text_splitter import split_text
from modules.vector_store import create_embeddings, save_index, load_index
from modules.rag_pipeline import generate_answer
from modules.question_generator import generate_questions
from modules.research_analyzer import analyze_paper
from modules.mindmap_generator import generate_mindmap
from modules.quiz_generator import generate_quiz, analyze_quiz_results
import warnings
import logging
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)


os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AcademIQ — AI Research Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "chat"
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True
if "index_ready" not in st.session_state:
    st.session_state.index_ready = False
# ── Quiz state ──
if "quiz_questions" not in st.session_state:
    st.session_state.quiz_questions = []
if "quiz_answers" not in st.session_state:
    st.session_state.quiz_answers = {}
if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False
if "quiz_results" not in st.session_state:
    st.session_state.quiz_results = None
if "quiz_live_answers" not in st.session_state:
    st.session_state.quiz_live_answers = {}
if "mindmap_data" not in st.session_state:
    st.session_state.mindmap_data = None
if "mindmap_chat" not in st.session_state:
    st.session_state.mindmap_chat = []

# ─────────────────────────────────────────────
# THEME
# ─────────────────────────────────────────────
dark = st.session_state.dark_mode

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown(f"""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&display=swap');

/* ── CSS Variables ── */
:root {{
  --bg:          {"#080c14" if dark else "#f0f4ff"};
  --bg2:         {"#0d1421" if dark else "#ffffff"};
  --bg3:         {"#111929" if dark else "#e8eef8"};
  --surface:     {"rgba(255,255,255,0.04)" if dark else "rgba(255,255,255,0.85)"};
  --surface2:    {"rgba(255,255,255,0.07)" if dark else "rgba(255,255,255,0.95)"};
  --border:      {"rgba(255,255,255,0.08)" if dark else "rgba(0,0,0,0.08)"};
  --text:        {"#e8edf5" if dark else "#0f172a"};
  --text2:       {"#8b95a8" if dark else "#64748b"};
  --text3:       {"#4a5568" if dark else "#94a3b8"};
  --accent:      #6366f1;
  --accent2:     #8b5cf6;
  --accent3:     #06b6d4;
  --glow:        rgba(99,102,241,0.25);
  --user-bubble: linear-gradient(135deg, #6366f1, #8b5cf6);
  --bot-bubble:  {"rgba(17,25,41,0.9)" if dark else "rgba(255,255,255,0.95)"};
  --shadow:      {"0 8px 32px rgba(0,0,0,0.4)" if dark else "0 8px 32px rgba(99,102,241,0.12)"};
  --font-head:   'Syne', sans-serif;
  --font-body:   'DM Sans', sans-serif;
  --radius:      16px;
  --radius-sm:   10px;
  --radius-xs:   6px;
  --sidebar-w:   260px;
}}

/* ── Reset & Base ── */
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

html, body, .stApp {{
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: var(--font-body) !important;
  font-size: 15px;
  line-height: 1.6;
}}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header,
.stDeployButton, [data-testid="stToolbar"],
[data-testid="stDecoration"] {{ display: none !important; }}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
  background: var(--bg2) !important;
  border-right: 1px solid var(--border) !important;
  width: var(--sidebar-w) !important;
}}
[data-testid="stSidebar"] > div:first-child {{
  padding: 0 !important;
}}
[data-testid="stSidebarContent"] {{
  padding: 0 !important;
}}

/* ── Main area ── */
.main .block-container {{
  max-width: 900px !important;
  padding: 2rem 2rem 8rem !important;
  margin: 0 auto !important;
}}

/* ── Scrollbar ── */
::-webkit-scrollbar {{ width: 5px; height: 5px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 99px; }}

/* ── Animations ── */
@keyframes fadeUp {{
  from {{ opacity: 0; transform: translateY(16px); }}
  to   {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes fadeIn {{
  from {{ opacity: 0; }}
  to   {{ opacity: 1; }}
}}
@keyframes pulse-dot {{
  0%, 80%, 100% {{ transform: scale(0.6); opacity: 0.4; }}
  40%           {{ transform: scale(1);   opacity: 1; }}
}}
@keyframes shimmer {{
  0%   {{ background-position: -400px 0; }}
  100% {{ background-position: 400px 0; }}
}}
@keyframes glow-pulse {{
  0%, 100% {{ box-shadow: 0 0 20px rgba(99,102,241,0.3); }}
  50%       {{ box-shadow: 0 0 40px rgba(99,102,241,0.6); }}
}}
@keyframes spin {{
  to {{ transform: rotate(360deg); }}
}}

/* ── Sidebar Logo Block ── */
.sb-logo {{
  padding: 24px 20px 16px;
  border-bottom: 1px solid var(--border);
  animation: fadeIn 0.4s ease;
}}
.sb-logo-name {{
  font-family: var(--font-head);
  font-size: 18px;
  font-weight: 800;
  background: linear-gradient(135deg, #6366f1, #a78bfa, #06b6d4);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  letter-spacing: -0.3px;
}}
.sb-logo-sub {{
  font-size: 11px;
  color: var(--text3);
  letter-spacing: 0.5px;
  text-transform: uppercase;
  margin-top: 2px;
}}

/* ── Sidebar Nav ── */
.sb-nav {{
  padding: 12px 12px 0;
}}
.sb-nav-label {{
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 1px;
  text-transform: uppercase;
  color: var(--text3);
  padding: 8px 8px 4px;
  margin-top: 8px;
}}
.sb-nav-item {{
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 12px;
  border-radius: var(--radius-sm);
  cursor: pointer;
  transition: all 0.2s ease;
  font-size: 14px;
  font-weight: 500;
  color: var(--text2);
  margin-bottom: 2px;
  text-decoration: none;
}}
.sb-nav-item:hover {{
  background: var(--surface);
  color: var(--text);
}}
.sb-nav-item.active {{
  background: linear-gradient(135deg, rgba(99,102,241,0.18), rgba(139,92,246,0.12));
  color: #a78bfa;
  border: 1px solid rgba(99,102,241,0.2);
}}
.sb-nav-item .icon {{
  width: 32px;
  height: 32px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 8px;
  font-size: 16px;
  background: var(--surface);
  flex-shrink: 0;
}}
.sb-nav-item.active .icon {{
  background: linear-gradient(135deg, #6366f1, #8b5cf6);
}}

/* ── Status Badge ── */
.status-badge {{
  display: inline-flex;
  align-items: center;
  gap: 5px;
  font-size: 11px;
  padding: 3px 8px;
  border-radius: 99px;
  font-weight: 500;
}}
.status-badge.ready {{
  background: rgba(16,185,129,0.15);
  color: #10b981;
  border: 1px solid rgba(16,185,129,0.25);
}}
.status-badge.offline {{
  background: rgba(239,68,68,0.12);
  color: #f87171;
  border: 1px solid rgba(239,68,68,0.2);
}}
.status-dot {{
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: currentColor;
}}

/* ── Page Header ── */
.page-header {{
  animation: fadeUp 0.5s ease;
  margin-bottom: 28px;
  padding-bottom: 20px;
  border-bottom: 1px solid var(--border);
}}
.page-title {{
  font-family: var(--font-head);
  font-size: 26px;
  font-weight: 800;
  color: var(--text);
  letter-spacing: -0.5px;
  line-height: 1.2;
}}
.page-title span {{
  background: linear-gradient(135deg, #6366f1, #a78bfa);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}}
.page-subtitle {{
  color: var(--text2);
  font-size: 14px;
  margin-top: 4px;
}}

/* ── Chat Messages ── */
.chat-wrap {{
  display: flex;
  flex-direction: column;
  gap: 18px;
  padding: 4px 0 24px;
  animation: fadeIn 0.3s ease;
}}
.msg-row {{
  display: flex;
  gap: 12px;
  animation: fadeUp 0.35s ease;
}}
.msg-row.user {{ flex-direction: row-reverse; }}

.avatar {{
  width: 36px;
  height: 36px;
  border-radius: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 16px;
  flex-shrink: 0;
  font-weight: 700;
}}
.avatar.ai {{
  background: linear-gradient(135deg, #6366f1, #8b5cf6);
  box-shadow: 0 0 0 2px rgba(99,102,241,0.25);
}}
.avatar.user {{
  background: linear-gradient(135deg, #0f172a, #1e293b);
  border: 1px solid var(--border);
  font-size: 13px;
  color: var(--text2);
}}

.bubble {{
  max-width: 72%;
  padding: 14px 18px;
  border-radius: var(--radius);
  font-size: 14.5px;
  line-height: 1.7;
  position: relative;
}}
.bubble.user {{
  background: var(--user-bubble);
  color: #fff;
  border-radius: var(--radius) 4px var(--radius) var(--radius);
  box-shadow: 0 4px 20px rgba(99,102,241,0.35);
}}
.bubble.ai {{
  background: var(--bot-bubble);
  color: var(--text);
  border: 1px solid var(--border);
  border-radius: 4px var(--radius) var(--radius) var(--radius);
  box-shadow: var(--shadow);
}}

.msg-meta {{
  font-size: 11px;
  color: var(--text3);
  margin-top: 6px;
  display: flex;
  align-items: center;
  gap: 8px;
}}
.msg-row.user .msg-meta {{ justify-content: flex-end; }}

/* ── Typing Indicator ── */
.typing-indicator {{
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 14px 18px;
  background: var(--bot-bubble);
  border: 1px solid var(--border);
  border-radius: 4px var(--radius) var(--radius) var(--radius);
  width: fit-content;
  box-shadow: var(--shadow);
}}
.typing-dot {{
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--accent);
  animation: pulse-dot 1.4s ease infinite;
}}
.typing-dot:nth-child(2) {{ animation-delay: 0.2s; }}
.typing-dot:nth-child(3) {{ animation-delay: 0.4s; }}

/* ── Empty State ── */
.empty-state {{
  text-align: center;
  padding: 60px 20px;
  animation: fadeUp 0.6s ease;
}}
.empty-icon {{
  font-size: 52px;
  margin-bottom: 16px;
  filter: drop-shadow(0 0 20px rgba(99,102,241,0.5));
}}
.empty-title {{
  font-family: var(--font-head);
  font-size: 22px;
  font-weight: 700;
  color: var(--text);
  margin-bottom: 8px;
}}
.empty-sub {{
  color: var(--text2);
  font-size: 14px;
  max-width: 420px;
  margin: 0 auto 32px;
}}
.suggestion-grid {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
  max-width: 520px;
  margin: 0 auto;
}}
.suggestion-chip {{
  padding: 12px 14px;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  cursor: pointer;
  font-size: 13px;
  color: var(--text2);
  transition: all 0.2s ease;
  text-align: left;
}}
.suggestion-chip:hover {{
  border-color: rgba(99,102,241,0.4);
  color: var(--text);
  background: var(--surface2);
}}

/* ── Source Card ── */
.source-card {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 12px 14px;
  font-size: 13px;
  color: var(--text2);
  line-height: 1.6;
  margin-bottom: 8px;
  border-left: 3px solid var(--accent);
  transition: all 0.2s ease;
}}
.source-card:hover {{
  border-color: var(--accent);
  color: var(--text);
  background: var(--surface2);
}}

/* ── Upload Area ── */
.upload-zone {{
  border: 2px dashed var(--border);
  border-radius: var(--radius);
  padding: 48px 24px;
  text-align: center;
  transition: all 0.3s ease;
  cursor: pointer;
  background: var(--surface);
  animation: fadeUp 0.4s ease;
}}
.upload-zone:hover {{
  border-color: var(--accent);
  background: rgba(99,102,241,0.05);
}}
.upload-icon {{ font-size: 40px; margin-bottom: 12px; }}
.upload-title {{
  font-family: var(--font-head);
  font-size: 17px;
  font-weight: 700;
  color: var(--text);
  margin-bottom: 6px;
}}
.upload-sub {{ font-size: 13px; color: var(--text2); }}

/* ── Cards ── */
.card {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 20px;
  margin-bottom: 12px;
  transition: all 0.25s ease;
  animation: fadeUp 0.4s ease;
}}
.card:hover {{
  border-color: rgba(99,102,241,0.25);
  box-shadow: 0 8px 24px rgba(99,102,241,0.1);
  transform: translateY(-1px);
}}
.card-title {{
  font-family: var(--font-head);
  font-size: 15px;
  font-weight: 700;
  color: var(--text);
  margin-bottom: 6px;
}}
.card-body {{
  font-size: 13.5px;
  color: var(--text2);
  line-height: 1.65;
}}

/* ── Info Grid ── */
.info-grid {{
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 12px;
  margin-bottom: 24px;
  animation: fadeUp 0.4s ease;
}}
.info-tile {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 16px;
  text-align: center;
}}
.info-tile-val {{
  font-family: var(--font-head);
  font-size: 26px;
  font-weight: 800;
  background: linear-gradient(135deg, #6366f1, #a78bfa);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}}
.info-tile-label {{
  font-size: 11px;
  color: var(--text3);
  margin-top: 2px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}}

/* ── Buttons ── */
.stButton > button {{
  font-family: var(--font-body) !important;
  font-weight: 600 !important;
  border-radius: var(--radius-sm) !important;
  border: none !important;
  transition: all 0.2s ease !important;
  cursor: pointer !important;
}}
.stButton > button[kind="primary"] {{
  background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
  color: white !important;
  box-shadow: 0 4px 16px rgba(99,102,241,0.35) !important;
}}
.stButton > button[kind="primary"]:hover {{
  transform: translateY(-1px) !important;
  box-shadow: 0 6px 24px rgba(99,102,241,0.5) !important;
}}
.stButton > button[kind="secondary"] {{
  background: var(--surface) !important;
  color: var(--text2) !important;
  border: 1px solid var(--border) !important;
}}
.stButton > button[kind="secondary"]:hover {{
  border-color: var(--accent) !important;
  color: var(--text) !important;
}}

/* ── Inputs ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {{
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--text) !important;
  font-family: var(--font-body) !important;
  transition: border-color 0.2s ease !important;
}}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {{
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px var(--glow) !important;
}}

/* ── Select Box ── */
.stSelectbox > div > div {{
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--text) !important;
}}

/* ── File Uploader ── */
[data-testid="stFileUploader"] {{
  background: var(--surface) !important;
  border: 2px dashed var(--border) !important;
  border-radius: var(--radius) !important;
  transition: all 0.2s ease !important;
}}
[data-testid="stFileUploader"]:hover {{
  border-color: var(--accent) !important;
}}
[data-testid="stFileUploader"] label {{
  color: var(--text2) !important;
}}

/* ── Expander ── */
.streamlit-expanderHeader {{
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--text2) !important;
  font-family: var(--font-body) !important;
  font-size: 13px !important;
}}
.streamlit-expanderContent {{
  background: var(--bg2) !important;
  border: 1px solid var(--border) !important;
  border-top: none !important;
  border-radius: 0 0 var(--radius-sm) var(--radius-sm) !important;
}}

/* ── Spinner override ── */
.stSpinner > div {{
  border-top-color: var(--accent) !important;
}}

/* ── Success / Warning / Error ── */
.stSuccess {{
  background: rgba(16,185,129,0.1) !important;
  border: 1px solid rgba(16,185,129,0.25) !important;
  border-radius: var(--radius-sm) !important;
  color: #34d399 !important;
}}
.stWarning {{
  background: rgba(245,158,11,0.1) !important;
  border: 1px solid rgba(245,158,11,0.25) !important;
  border-radius: var(--radius-sm) !important;
}}
.stError {{
  background: rgba(239,68,68,0.1) !important;
  border: 1px solid rgba(239,68,68,0.25) !important;
  border-radius: var(--radius-sm) !important;
}}

/* ── Divider ── */
hr {{
  border: none !important;
  border-top: 1px solid var(--border) !important;
  margin: 20px 0 !important;
}}

/* ── Progress Bar ── */
.stProgress > div > div > div {{
  background: linear-gradient(90deg, #6366f1, #8b5cf6) !important;
  border-radius: 99px !important;
}}

/* ── Toggle ── */
.stToggle {{
  color: var(--text2) !important;
}}

/* ── Tabs (hidden — we use sidebar navigation) ── */
.stTabs [data-baseweb="tab-list"] {{
  background: var(--bg2) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
  padding: 4px !important;
  gap: 4px !important;
}}
.stTabs [data-baseweb="tab"] {{
  background: transparent !important;
  color: var(--text2) !important;
  border-radius: var(--radius-xs) !important;
  font-family: var(--font-body) !important;
  font-size: 13.5px !important;
  font-weight: 500 !important;
  border: none !important;
  padding: 8px 16px !important;
}}
.stTabs [aria-selected="true"] {{
  background: linear-gradient(135deg, rgba(99,102,241,0.2), rgba(139,92,246,0.15)) !important;
  color: #a78bfa !important;
}}

/* ── Responsive ── */
@media (max-width: 768px) {{
  .main .block-container {{
    padding: 1rem 1rem 6rem !important;
  }}
  .info-grid {{ grid-template-columns: 1fr 1fr; }}
  .suggestion-grid {{ grid-template-columns: 1fr; }}
  .bubble {{ max-width: 88%; }}
}}
/* ── Quiz UI ── */
.quiz-header {{
  animation: fadeUp 0.5s ease;
  margin-bottom: 24px;
}}
.q-card {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 22px 24px;
  margin-bottom: 16px;
  transition: border-color 0.2s ease;
  animation: fadeUp 0.4s ease;
}}
.q-card:hover {{ border-color: rgba(99,102,241,0.3); }}
.q-badge {{
  display: inline-flex;
  align-items: center;
  gap: 5px;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.8px;
  text-transform: uppercase;
  padding: 3px 9px;
  border-radius: 99px;
  margin-bottom: 10px;
}}
.q-badge.mcq      {{ background: rgba(99,102,241,0.15); color:#a78bfa; border:1px solid rgba(99,102,241,0.25); }}
.q-badge.truefalse{{ background: rgba(6,182,212,0.15);  color:#67e8f9; border:1px solid rgba(6,182,212,0.25); }}
.q-badge.short    {{ background: rgba(16,185,129,0.15); color:#6ee7b7; border:1px solid rgba(16,185,129,0.25); }}
.q-text {{
  font-family: var(--font-head);
  font-size: 15.5px;
  font-weight: 600;
  color: var(--text);
  margin-bottom: 14px;
  line-height: 1.5;
}}
.q-topic {{
  font-size: 11px;
  color: var(--text3);
  margin-bottom: 12px;
  display: flex;
  align-items: center;
  gap: 4px;
}}

/* Results */
.result-hero {{
  text-align: center;
  padding: 40px 20px 28px;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  margin-bottom: 24px;
  animation: fadeUp 0.5s ease;
}}
.score-ring {{
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 110px;
  height: 110px;
  border-radius: 50%;
  font-family: var(--font-head);
  font-size: 30px;
  font-weight: 800;
  margin-bottom: 14px;
  position: relative;
}}
.score-ring.great {{ background: radial-gradient(circle, rgba(16,185,129,0.2), rgba(16,185,129,0.05)); border: 3px solid #10b981; color: #34d399; box-shadow: 0 0 30px rgba(16,185,129,0.3); }}
.score-ring.ok    {{ background: radial-gradient(circle, rgba(245,158,11,0.2), rgba(245,158,11,0.05)); border: 3px solid #f59e0b; color: #fbbf24; box-shadow: 0 0 30px rgba(245,158,11,0.3); }}
.score-ring.low   {{ background: radial-gradient(circle, rgba(239,68,68,0.2),  rgba(239,68,68,0.05));  border: 3px solid #ef4444; color: #f87171; box-shadow: 0 0 30px rgba(239,68,68,0.3); }}
.score-label {{
  font-family: var(--font-head);
  font-size: 20px;
  font-weight: 700;
  color: var(--text);
  margin-bottom: 4px;
}}
.score-sub {{ font-size: 13px; color: var(--text2); }}

.topic-bar-row {{
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 10px;
  font-size: 13px;
}}
.topic-bar-label {{ width: 160px; flex-shrink:0; color: var(--text2); font-weight:500; }}
.topic-bar-track {{
  flex: 1;
  height: 8px;
  background: var(--surface2);
  border-radius: 99px;
  overflow: hidden;
  border: 1px solid var(--border);
}}
.topic-bar-fill {{
  height: 100%;
  border-radius: 99px;
  transition: width 0.8s ease;
}}
.topic-bar-pct {{ width: 36px; text-align:right; color:var(--text3); font-size:12px; }}

.insight-row {{
  display: flex;
  align-items: flex-start;
  gap: 10px;
  padding: 10px 14px;
  border-radius: var(--radius-sm);
  margin-bottom: 8px;
  font-size: 13.5px;
  animation: fadeUp 0.4s ease;
}}
.insight-row.strong {{ background:rgba(16,185,129,0.08); border:1px solid rgba(16,185,129,0.2); color:#6ee7b7; }}
.insight-row.weak   {{ background:rgba(239,68,68,0.08);  border:1px solid rgba(239,68,68,0.2);  color:#fca5a5; }}
.insight-row.rec    {{ background:rgba(99,102,241,0.08); border:1px solid rgba(99,102,241,0.2); color:#c4b5fd; }}

.ans-correct {{ color: #34d399 !important; font-weight:600; }}
.ans-wrong   {{ color: #f87171 !important; font-weight:600; }}

/* progress bar for quiz */
.quiz-progress-bar {{
  height: 4px;
  background: var(--surface2);
  border-radius: 99px;
  margin-bottom: 24px;
  overflow: hidden;
}}
.quiz-progress-fill {{
  height: 100%;
  background: linear-gradient(90deg, #6366f1, #8b5cf6);
  border-radius: 99px;
  transition: width 0.4s ease;
}}

</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def format_time():
    return time.strftime("%I:%M %p")


def render_chat_message(role: str, content: str, ts: str = ""):
    if role == "user":
        st.markdown(f"""
        <div class="msg-row user">
          <div class="avatar user">You</div>
          <div>
            <div class="bubble user">{content}</div>
            <div class="msg-meta">{ts}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="msg-row">
          <div class="avatar ai">🎓</div>
          <div>
            <div class="bubble ai">{content}</div>
            <div class="msg-meta">{ts}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)


def render_typing():
    st.markdown("""
    <div class="msg-row">
      <div class="avatar ai">🎓</div>
      <div class="typing-indicator">
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:

    # Logo
    st.markdown("""
    <div class="sb-logo">
      <div style="font-size:28px;margin-bottom:6px;">🎓</div>
      <div class="sb-logo-name">AcademIQ</div>
      <div class="sb-logo-sub">AI Research Assistant</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-nav">', unsafe_allow_html=True)

    # ── Nav: Main ──
    st.markdown('<div class="sb-nav-label">Workspace</div>', unsafe_allow_html=True)

    pages = [
        ("chat",      "💬", "Chat"),
        ("upload",    "📄", "Documents"),
        ("mindmap",    "📄", "Mindmap"),
        ("quiz",      "🧠", "Quiz"),
        ("questions", "❓", "Questions"),
        ("analyzer",  "🔬", "Analyzer"),
    ]
    for key, icon, label in pages:
        active = "active" if st.session_state.active_tab == key else ""
        if st.button(f"{icon}  {label}", key=f"nav_{key}",
                     use_container_width=True,
                     help=f"Go to {label}"):
            st.session_state.active_tab = key
            st.rerun()

    st.markdown("---")

    # ── Document Manager ──
    st.markdown('<div class="sb-nav-label">Document Manager</div>', unsafe_allow_html=True)

    files = st.file_uploader("📎 Upload Files", accept_multiple_files=True,
                             type=["pdf", "docx", "txt"],
                             label_visibility="collapsed")
    url_input = st.text_input("🔗 Enter URL", placeholder="https://arxiv.org/...",
                              label_visibility="collapsed")

    col1, col2 = st.columns(2)
    with col1:
        process_btn = st.button("⚡ Process", use_container_width=True, type="primary")
    with col2:
        clear_btn   = st.button("🗑️ Clear",   use_container_width=True)

    if process_btn:
        text_data = ""
        if files:
            for file in files:
                text_data += load_document(file)
        if url_input:
            text_data += load_url(url_input)

        if text_data.strip():
            with st.spinner("Building knowledge base…"):
                chunks = split_text(text_data)
                chunks = chunks[:20]
                embeddings = create_embeddings(chunks)
                save_index(embeddings, chunks)
                st.session_state.index_ready = True
            st.success("✅ Ready for Chat!")
        else:
            st.warning("No input provided")

    if clear_btn:
        for p in ["data/faiss_index/index.faiss", "data/faiss_index/texts.pkl"]:
            if os.path.exists(p):
                os.remove(p)
        st.session_state.index_ready = False
        st.success("Index cleared")

    # ── Index status badge ──
    idx, _ = load_index()
    is_ready = idx is not None
    badge_cls  = "ready" if is_ready  else "offline"
    badge_text = "Index Ready" if is_ready else "No Index"
    st.markdown(f"""
    <div style="padding:6px 0 4px;">
      <span class="status-badge {badge_cls}">
        <span class="status-dot"></span>{badge_text}
      </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Voice Input ──
    st.markdown('<div class="sb-nav-label">🎤 Voice Input</div>', unsafe_allow_html=True)
    audio_file = st.file_uploader("Upload Voice (wav/mp3)", type=["wav", "mp3"],
                                  label_visibility="collapsed")
    voice_text = ""
    if audio_file is not None:
        from modules.audio_loader import transcribe_audio
        with st.spinner("Transcribing… 🎧"):
            voice_text = transcribe_audio(audio_file)
        st.success("✅ Voice converted")
        st.caption(f"_{voice_text[:120]}…_" if len(voice_text) > 120 else f"_{voice_text}_")

    st.markdown("---")

    # ── Theme toggle ──
    dark_toggle = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)
    if dark_toggle != st.session_state.dark_mode:
        st.session_state.dark_mode = dark_toggle
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# ╔═══════════════════════════════╗
# ║        MAIN CONTENT           ║
# ╚═══════════════════════════════╝
# ─────────────────────────────────────────────
tab = st.session_state.active_tab


# ══════════════════════════════════════════════
# 💬  CHAT
# ══════════════════════════════════════════════
if tab == "chat":

    st.markdown("""
    <div class="page-header">
      <div class="page-title">AI <span>Chat</span></div>
      <div class="page-subtitle">Ask questions grounded in your uploaded documents</div>
    </div>
    """, unsafe_allow_html=True)

    # Stats row
    total_q = len(st.session_state.chat_history)
    idx_chk, idx_texts = load_index()
    doc_chunks = len(idx_texts) if idx_texts else 0

    st.markdown(f"""
    <div class="info-grid">
      <div class="info-tile">
        <div class="info-tile-val">{total_q}</div>
        <div class="info-tile-label">Questions Asked</div>
      </div>
      <div class="info-tile">
        <div class="info-tile-val">{doc_chunks}</div>
        <div class="info-tile-label">Doc Chunks Indexed</div>
      </div>
      <div class="info-tile">
        <div class="info-tile-val">{"On" if idx_chk else "Off"}</div>
        <div class="info-tile-label">RAG Status</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Chat history display
    if not st.session_state.chat_history:
        st.markdown("""
        <div class="empty-state">
          <div class="empty-icon">🤖</div>
          <div class="empty-title">How can I help you research?</div>
          <div class="empty-sub">Upload a document or paste a URL in the sidebar, then ask anything about it.</div>
          <div class="suggestion-grid">
            <div class="suggestion-chip">📌 Summarize the key findings</div>
            <div class="suggestion-chip">🔍 What methodology was used?</div>
            <div class="suggestion-chip">📊 List the main results</div>
            <div class="suggestion-chip">💡 Explain the core concepts</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)
        for item in st.session_state.chat_history:
            render_chat_message("user", item["q"], item.get("ts_q", ""))
            if "a" in item:
                render_chat_message("ai", item["a"].replace("\n", "<br>"), item.get("ts_a", ""))
        st.markdown("</div>", unsafe_allow_html=True)

    # Chat input
    query = st.chat_input("Ask anything from your documents…")
    if voice_text:
        query = voice_text

    if query:
        index, texts = load_index()
        if index is None:
            st.warning("⚠️ Please process documents first — use the sidebar to upload files.")
            st.stop()

        ts_q = format_time()
        render_chat_message("user", query, ts_q)

        typing_placeholder = st.empty()
        with typing_placeholder.container():
            render_typing()

        with st.spinner(""):
            answer, chunks = generate_answer(query, index, texts)

        typing_placeholder.empty()
        ts_a = format_time()
        render_chat_message("ai", answer.replace("\n", "<br>"), ts_a)

        st.session_state.chat_history.append({
            "q": query, "a": answer,
            "ts_q": ts_q, "ts_a": ts_a,
        })

        if chunks:
            with st.expander("📚 Source Context Used", expanded=False):
                for i, c in enumerate(chunks):
                    st.markdown(f"""
                    <div class="source-card">
                      <span style="font-size:11px;color:var(--accent);font-weight:600;
                                   text-transform:uppercase;letter-spacing:0.5px;">
                        Source {i+1}
                      </span><br>
                      {c}
                    </div>
                    """, unsafe_allow_html=True)

    # Clear chat
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()


# ══════════════════════════════════════════════
# 📄  DOCUMENTS
# ══════════════════════════════════════════════
elif tab == "upload":

    st.markdown("""
    <div class="page-header">
      <div class="page-title">Document <span>Manager</span></div>
      <div class="page-subtitle">Upload, process, and manage your academic documents</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="upload-zone">
      <div class="upload-icon">📁</div>
      <div class="upload-title">Drop documents or use the sidebar uploader</div>
      <div class="upload-sub">Supports PDF, DOCX, TXT — or paste a URL below</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        <div class="card">
          <div class="card-title">📋 Supported Formats</div>
          <div class="card-body">
            <b>PDF</b> — Research papers, textbooks<br>
            <b>DOCX</b> — Word documents<br>
            <b>TXT</b> — Plain text files<br>
            <b>URL</b> — Web pages & online papers
          </div>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        idx, txts = load_index()
        chunk_count = len(txts) if txts else 0
        st.markdown(f"""
        <div class="card">
          <div class="card-title">📊 Index Status</div>
          <div class="card-body">
            <b>Chunks indexed:</b> {chunk_count}<br>
            <b>Status:</b> {"✅ Ready" if idx else "⚠️ No index"}<br>
            <b>Max chunks:</b> 20 (speed optimised)<br>
            <b>Engine:</b> FAISS + Sentence Transformers
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
      <div class="card-title">🚀 How to get started</div>
      <div class="card-body">
        1. Upload files via <b>sidebar → Document Manager</b><br>
        2. Optionally add a <b>URL</b> of an online paper<br>
        3. Click <b>⚡ Process</b> to build the knowledge index<br>
        4. Switch to <b>💬 Chat</b> to ask questions
      </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# ❓  QUESTION GENERATOR
# ══════════════════════════════════════════════
elif tab == "questions":

    st.markdown("""
    <div class="page-header">
      <div class="page-title">Question <span>Generator</span></div>
      <div class="page-subtitle">Auto-generate academic questions from your indexed documents</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
      <div class="card-title">⚙️ Generation Settings</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        difficulty = st.selectbox(
            "Difficulty Level",
            ["Easy", "Medium", "Hard"],
            help="Controls question complexity"
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        gen_btn = st.button("✨ Generate Questions", type="primary", use_container_width=True)

    if gen_btn:
        index, texts = load_index()
        if index is None:
            st.warning("⚠️ Please process documents first")
            st.stop()

        text = " ".join(texts)
        with st.spinner(f"Generating {difficulty} questions…"):
            result = generate_questions(text, difficulty)

        st.markdown(f"""
        <div class="card" style="border-left:3px solid var(--accent);">
          <div class="card-title">📝 Generated Questions — {difficulty}</div>
          <div class="card-body" style="white-space:pre-wrap;">{result}</div>
        </div>
        """, unsafe_allow_html=True)

        st.download_button(
            "⬇️ Download as TXT",
            data=result,
            file_name=f"questions_{difficulty.lower()}.txt",
            mime="text/plain",
        )


# ══════════════════════════════════════════════
# 🔬  ANALYZER
# ══════════════════════════════════════════════
elif tab == "analyzer":

    st.markdown("""
    <div class="page-header">
      <div class="page-title">Research <span>Analyzer</span></div>
      <div class="page-subtitle">Deep analysis and insights from your academic documents</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
      <div class="card-title">🔬 Paper Analysis Engine</div>
      <div class="card-body">
        Extracts key insights, methodology, findings, and conclusions from your indexed paper.
      </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🚀 Analyze Document", type="primary"):
        index, texts = load_index()
        if index is None:
            st.warning("⚠️ Please process documents first")
            st.stop()

        text = " ".join(texts)
        with st.spinner("Analyzing research paper… this may take a moment"):
            result = analyze_paper(text)

        st.markdown(f"""
        <div class="card" style="border-left:3px solid var(--accent3);">
          <div class="card-title">📄 Analysis Report</div>
          <div class="card-body" style="white-space:pre-wrap;">{result}</div>
        </div>
        """, unsafe_allow_html=True)

        st.download_button(
            "⬇️ Download Report",
            data=result,
            file_name="analysis_report.txt",
            mime="text/plain",
        )

# ══════════════════════════════════════════════
# 🧠  QUIZ
# ══════════════════════════════════════════════
elif tab == "quiz":

    st.markdown("""
    <div class="page-header">
      <div class="page-title">🧠 Knowledge <span>Quiz</span></div>
      <div class="page-subtitle">MCQ &amp; True/False questions generated directly from your document</div>
    </div>
    """, unsafe_allow_html=True)

    # ────────────────────────────────────────────
    # PHASE 1 — GENERATE
    # ────────────────────────────────────────────
    if not st.session_state.quiz_questions:

        index_chk, _ = load_index()

        if index_chk is None:
            st.markdown("""
            <div class="card" style="border-left:3px solid #f59e0b;text-align:center;padding:32px;">
              <div style="font-size:36px;margin-bottom:10px;">⚠️</div>
              <div class="card-title">No document indexed yet</div>
              <div class="card-body">Upload a PDF, DOCX, TXT or URL in the sidebar and click <b>⚡ Process</b> first.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="card">
              <div class="card-title">📋 About this Quiz</div>
              <div class="card-body">
                Questions are generated <b>directly from your uploaded document</b> —
                every question tests real content from the text, not generic knowledge.<br><br>
                <b>Format:</b> 7 Multiple Choice Questions + 5 True / False statements<br>
                <b>Scoring:</b> Instant result with per-topic breakdown and study advice
              </div>
            </div>
            """, unsafe_allow_html=True)

            if st.button("🚀 Generate Quiz from Document", type="primary", use_container_width=True):
                _, texts = load_index()
                full_text = " ".join(texts)

                with st.spinner("🧠 Reading your document and writing questions…"):
                    questions = generate_quiz(full_text)

                if not questions:
                    st.error(
                        "❌ Could not generate questions from this document. "
                        "The LLM may not have returned valid JSON. "
                        "Try re-processing the document or uploading a longer text."
                    )
                else:
                    st.session_state.quiz_questions    = questions
                    st.session_state.quiz_submitted    = False
                    st.session_state.quiz_answers      = {}
                    st.session_state.quiz_results      = None
                    st.session_state.quiz_live_answers = {}
                    st.rerun()

    # ────────────────────────────────────────────
    # PHASE 2 — TAKE THE QUIZ
    # ────────────────────────────────────────────
    elif not st.session_state.quiz_submitted:

        questions = st.session_state.quiz_questions
        total_q   = len(questions)

        # Live progress
        answered = sum(
            1 for i in range(total_q)
            if st.session_state.quiz_live_answers.get(i) not in (None, "")
        )
        pct_done = int(answered / total_q * 100) if total_q else 0

        mcq_count = sum(1 for q in questions if q["type"] == "mcq")
        tf_count  = sum(1 for q in questions if q["type"] == "truefalse")

        st.markdown(f"""
        <div style="display:flex;align-items:center;justify-content:space-between;
                    margin-bottom:6px;font-size:13px;color:var(--text2);">
          <span>📝 {mcq_count} MCQ &nbsp;|&nbsp; {tf_count} True/False &nbsp;·&nbsp; {total_q} questions total</span>
          <span style="color:var(--accent);font-weight:600;">{answered}/{total_q} answered</span>
        </div>
        <div class="quiz-progress-bar">
          <div class="quiz-progress-fill" style="width:{pct_done}%;"></div>
        </div>
        """, unsafe_allow_html=True)

        # Render questions
        for i, q in enumerate(questions):
            q_type  = q.get("type", "mcq")
            topic   = q.get("topic", "General")
            q_text  = q.get("question", "")
            options = q.get("options", [])
            badge   = "MCQ" if q_type == "mcq" else "True / False"

            st.markdown(f"""
            <div class="q-card">
              <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;">
                <span class="q-badge {q_type}">{badge}</span>
                <span style="font-size:11px;color:var(--text3);">📌 {topic}</span>
                <span style="margin-left:auto;font-size:11px;color:var(--text3);">Q{i+1} of {total_q}</span>
              </div>
              <div class="q-text">{q_text}</div>
            </div>
            """, unsafe_allow_html=True)

            if q_type == "mcq":
                prev = st.session_state.quiz_live_answers.get(i)
                prev_idx = None
                if prev:
                    import re as _re2
                    m = _re2.match(r"([A-Da-d])", prev)
                    if m:
                        letter = m.group(1).upper()
                        for oi, opt in enumerate(options):
                            if opt.startswith(letter):
                                prev_idx = oi
                                break

                choice = st.radio(
                    f"q{i}",
                    options=options,
                    key=f"quiz_q_{i}",
                    index=prev_idx,
                    label_visibility="collapsed",
                )
                if choice:
                    import re as _re3
                    m2 = _re3.match(r"([A-Da-d])", choice)
                    if m2:
                        st.session_state.quiz_live_answers[i] = m2.group(1).upper()

            else:  # truefalse
                prev_tf = st.session_state.quiz_live_answers.get(i)
                tf_idx  = None
                if prev_tf == "True":  tf_idx = 0
                if prev_tf == "False": tf_idx = 1

                tf_choice = st.radio(
                    f"q{i}",
                    options=["True", "False"],
                    key=f"quiz_q_{i}",
                    index=tf_idx,
                    label_visibility="collapsed",
                    horizontal=True,
                )
                if tf_choice:
                    st.session_state.quiz_live_answers[i] = tf_choice

            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Submit / Reset
        col_sub, col_rst = st.columns([2, 1])
        with col_sub:
            submit_disabled = answered < total_q
            if st.button(
                f"✅ Submit Quiz  ({answered}/{total_q} answered)",
                type="primary",
                use_container_width=True,
                disabled=submit_disabled,
            ):
                st.session_state.quiz_answers   = dict(st.session_state.quiz_live_answers)
                st.session_state.quiz_submitted = True
                st.session_state.quiz_results   = analyze_quiz_results(
                    st.session_state.quiz_questions,
                    st.session_state.quiz_answers,
                )
                st.rerun()
        with col_rst:
            if st.button("🔄 New Quiz", use_container_width=True):
                st.session_state.quiz_questions    = []
                st.session_state.quiz_live_answers = {}
                st.rerun()

        if submit_disabled:
            st.caption(f"Answer all {total_q} questions to submit.")

    # ────────────────────────────────────────────
    # PHASE 3 — RESULTS
    # ────────────────────────────────────────────
    else:
        import re as _re
        results   = st.session_state.quiz_results
        questions = st.session_state.quiz_questions

        score_pct = results["score_pct"]
        correct   = results["correct"]
        total     = results["total"]
        topic_scores      = results["topic_scores"]
        strongest         = results["strongest_topics"]
        weakest           = results["weakest_topics"]
        recommendations   = results["recommendations"]
        per_q             = results["per_question"]

        ring_cls  = "great" if score_pct >= 75 else ("ok" if score_pct >= 50 else "low")
        grade_msg = (
            "🌟 Outstanding!"        if score_pct >= 90 else
            "🎉 Great Work!"         if score_pct >= 75 else
            "📚 Keep Studying"       if score_pct >= 50 else
            "🔄 Needs More Revision"
        )

        # ── Score hero ──
        st.markdown(f"""
        <div class="result-hero">
          <div class="score-ring {ring_cls}">{score_pct}%</div>
          <div class="score-label">{grade_msg}</div>
          <div class="score-sub">{correct} correct out of {total} questions</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Topic breakdown + Strengths/Weaknesses ──
        col_l, col_r = st.columns([3, 2])

        with col_l:
            st.markdown("""
            <div style="font-family:var(--font-head);font-size:15px;font-weight:700;
                        color:var(--text);margin-bottom:14px;">📊 Topic Performance</div>
            """, unsafe_allow_html=True)

            for topic, ts in sorted(topic_scores.items(), key=lambda x: x[1]["pct"], reverse=True):
                tp = ts["pct"]
                correct_t = ts["correct"]
                total_t   = ts["total"]
                bar_color = (
                    "linear-gradient(90deg,#10b981,#34d399)" if tp >= 70 else
                    "linear-gradient(90deg,#f59e0b,#fbbf24)" if tp >= 40 else
                    "linear-gradient(90deg,#ef4444,#f87171)"
                )
                st.markdown(f"""
                <div class="topic-bar-row">
                  <div class="topic-bar-label" title="{topic}">{topic[:22] + "…" if len(topic) > 22 else topic}</div>
                  <div class="topic-bar-track">
                    <div class="topic-bar-fill" style="width:{tp}%;background:{bar_color};"></div>
                  </div>
                  <div class="topic-bar-pct">{correct_t}/{total_t}</div>
                </div>
                """, unsafe_allow_html=True)

        with col_r:
            if strongest:
                st.markdown("""
                <div style="font-family:var(--font-head);font-size:14px;font-weight:700;
                            color:var(--text);margin-bottom:8px;">💪 Strongest Topics</div>
                """, unsafe_allow_html=True)
                for t in strongest:
                    pct = topic_scores[t]["pct"]
                    st.markdown(
                        f'<div class="insight-row strong">✅ {t} &nbsp;<b>{pct}%</b></div>',
                        unsafe_allow_html=True,
                    )

            if weakest:
                st.markdown("""
                <div style="font-family:var(--font-head);font-size:14px;font-weight:700;
                            color:var(--text);margin:14px 0 8px;">⚠️ Needs Work</div>
                """, unsafe_allow_html=True)
                for t in weakest:
                    pct = topic_scores[t]["pct"]
                    st.markdown(
                        f'<div class="insight-row weak">📌 {t} &nbsp;<b>{pct}%</b></div>',
                        unsafe_allow_html=True,
                    )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Study Recommendations ──
        st.markdown("""
        <div style="font-family:var(--font-head);font-size:15px;font-weight:700;
                    color:var(--text);margin-bottom:10px;">🎯 What You Should Focus On</div>
        """, unsafe_allow_html=True)
        for rec in recommendations:
            st.markdown(f'<div class="insight-row rec">💡 {rec}</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Detailed Q&A Review ──
        st.markdown("""
        <div style="font-family:var(--font-head);font-size:15px;font-weight:700;
                    color:var(--text);margin-bottom:12px;">📝 Full Answer Review</div>
        """, unsafe_allow_html=True)

        for i, pq in enumerate(per_q):
            is_correct  = pq["is_correct"]
            q_type      = pq["type"]
            icon        = "✅" if is_correct else "❌"
            badge_label = "MCQ" if q_type == "mcq" else "T/F"
            result_color = "#34d399" if is_correct else "#f87171"
            result_text  = "Correct" if is_correct else "Incorrect"

            # Build the options block for MCQ
            options_html = ""
            if q_type == "mcq":
                for opt in pq.get("options", []):
                    letter = opt[:1].upper()
                    is_right  = letter == pq["correct_ans"].upper()
                    is_chosen = letter == pq["user_ans"].upper()
                    opt_style = ""
                    opt_icon  = ""
                    if is_right:
                        opt_style = "color:#34d399;font-weight:600;"
                        opt_icon  = " ✓"
                    elif is_chosen and not is_right:
                        opt_style = "color:#f87171;text-decoration:line-through;"
                        opt_icon  = " ✗"
                    options_html += f'<div style="font-size:13px;padding:3px 0;{opt_style}">{opt}{opt_icon}</div>'

            tf_html = ""
            if q_type == "truefalse":
                correct_ans = pq["correct_ans"]
                user_ans    = pq["user_ans"]
                for tf_opt in ["True", "False"]:
                    is_right  = tf_opt == correct_ans
                    is_chosen = tf_opt == user_ans
                    opt_style = ""
                    opt_icon  = ""
                    if is_right:
                        opt_style = "color:#34d399;font-weight:600;"
                        opt_icon  = " ✓"
                    elif is_chosen and not is_right:
                        opt_style = "color:#f87171;text-decoration:line-through;"
                        opt_icon  = " ✗"
                    tf_html += f'<div style="font-size:13px;padding:3px 0;{opt_style}">{tf_opt}{opt_icon}</div>'

            expl_html = ""
            if pq.get("explanation"):
                expl_html = f"""
                <div style="margin-top:10px;padding-top:10px;border-top:1px solid var(--border);
                             font-size:12.5px;color:var(--text3);">
                  💡 {pq['explanation']}
                </div>"""

            st.markdown(f"""
            <div class="q-card" style="border-left:3px solid {result_color};">
              <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
                <span class="q-badge {q_type}">{badge_label}</span>
                <span style="font-size:11px;color:var(--text3);">📌 {pq['topic']}</span>
                <span style="margin-left:auto;font-size:13px;font-weight:700;color:{result_color};">
                  {icon} {result_text}
                </span>
              </div>
              <div class="q-text" style="font-size:14px;margin-bottom:10px;">
                Q{i+1}. {pq['question']}
              </div>
              {options_html}
              {tf_html}
              {expl_html}
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Action Buttons ──
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🔄 Retake Same Quiz", type="primary", use_container_width=True):
                st.session_state.quiz_submitted    = False
                st.session_state.quiz_answers      = {}
                st.session_state.quiz_results      = None
                st.session_state.quiz_live_answers = {}
                st.rerun()
        with col_b:
            if st.button("🆕 Generate New Quiz", use_container_width=True):
                st.session_state.quiz_questions    = []
                st.session_state.quiz_submitted    = False
                st.session_state.quiz_answers      = {}
                st.session_state.quiz_results      = None
                st.session_state.quiz_live_answers = {}
                st.rerun()


# ══════════════════════════════════════════════
#  🗺️  MIND MAP
# ══════════════════════════════════════════════
elif tab == "mindmap":

    # ── Extra CSS for the Mind Map page ──────────────────────────────────────
    st.markdown("""
    <style>
    /* Hero */
    .mm-hero {
        text-align: center;
        padding: 48px 24px 36px;
        background: linear-gradient(135deg,
            rgba(99,102,241,0.08) 0%,
            rgba(139,92,246,0.05) 50%,
            rgba(6,182,212,0.06) 100%);
        border: 1px solid rgba(99,102,241,0.18);
        border-radius: 20px;
        margin-bottom: 32px;
        position: relative;
        overflow: hidden;
        animation: fadeUp 0.6s ease;
    }
    .mm-hero::before {
        content: "";
        position: absolute;
        inset: 0;
        background: radial-gradient(ellipse at 50% 0%,
            rgba(99,102,241,0.18) 0%, transparent 70%);
        pointer-events: none;
    }
    .mm-ai-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: linear-gradient(135deg,
            rgba(99,102,241,0.2), rgba(139,92,246,0.15));
        border: 1px solid rgba(99,102,241,0.3);
        border-radius: 99px;
        padding: 5px 14px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        color: #a78bfa;
        margin-bottom: 18px;
    }
    .mm-hero-title {
        font-family: var(--font-head);
        font-size: 38px;
        font-weight: 800;
        letter-spacing: -1px;
        line-height: 1.15;
        margin-bottom: 12px;
        background: linear-gradient(135deg, #e8edf5, #a78bfa, #67e8f9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .mm-hero-sub {
        font-size: 15px;
        color: var(--text2);
        max-width: 520px;
        margin: 0 auto 24px;
        line-height: 1.65;
    }
    .mm-chips {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 10px;
    }
    .mm-chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 8px 16px;
        border-radius: 99px;
        font-size: 13px;
        font-weight: 600;
        cursor: default;
        transition: all 0.25s ease;
        border: 1px solid;
    }
    .mm-chip:nth-child(1) { background: rgba(99,102,241,0.12); border-color: rgba(99,102,241,0.3); color:#a78bfa; }
    .mm-chip:nth-child(2) { background: rgba(245,158,11,0.12); border-color: rgba(245,158,11,0.3); color:#fbbf24; }
    .mm-chip:nth-child(3) { background: rgba(6,182,212,0.12);  border-color: rgba(6,182,212,0.3);  color:#67e8f9; }
    .mm-chip:nth-child(4) { background: rgba(16,185,129,0.12); border-color: rgba(16,185,129,0.3); color:#6ee7b7; }
    .mm-chip:hover { transform: translateY(-2px); box-shadow: 0 6px 18px rgba(0,0,0,0.25); }

    /* Panels */
    .mm-panel {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 24px;
        margin-bottom: 20px;
        animation: fadeUp 0.4s ease;
        transition: border-color 0.25s;
    }
    .mm-panel:hover { border-color: rgba(99,102,241,0.25); }
    .mm-panel-title {
        font-family: var(--font-head);
        font-size: 16px;
        font-weight: 700;
        color: var(--text);
        margin-bottom: 4px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .mm-panel-sub { font-size: 13px; color: var(--text2); margin-bottom: 16px; }

    /* Chat bubbles */
    .mm-chat-wrap { display:flex; flex-direction:column; gap:14px; padding: 4px 0 16px; }
    .mm-msg-row   { display:flex; gap:10px; animation: fadeUp 0.3s ease; }
    .mm-msg-row.user { flex-direction:row-reverse; }
    .mm-bubble { max-width:76%; padding:12px 16px; border-radius:14px; font-size:14px; line-height:1.65; }
    .mm-bubble.user { background:var(--user-bubble); color:#fff; border-radius:14px 4px 14px 14px; }
    .mm-bubble.ai   { background:var(--bot-bubble); border:1px solid var(--border); color:var(--text); border-radius:4px 14px 14px 14px; }

    /* Viz container */
    .mm-viz-wrap {
        width: 100%;
        background: radial-gradient(ellipse at center,
            rgba(99,102,241,0.06) 0%, rgba(8,12,20,0.0) 70%),
            var(--bg2);
        border: 1px solid var(--border);
        border-radius: 20px;
        overflow: hidden;
        position: relative;
        animation: fadeIn 0.5s ease;
    }
    .mm-viz-toolbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 12px 18px;
        border-bottom: 1px solid var(--border);
        background: var(--surface);
    }
    .mm-viz-title { font-family:var(--font-head); font-size:14px; font-weight:700; color:var(--text); }
    .mm-viz-hint  { font-size:12px; color:var(--text3); }

    /* Error card */
    .mm-error {
        background: rgba(239,68,68,0.08);
        border: 1px solid rgba(239,68,68,0.25);
        border-radius: var(--radius);
        padding: 18px 20px;
        color: #fca5a5;
        font-size: 14px;
        line-height: 1.65;
        animation: fadeUp 0.4s ease;
    }
    .mm-error strong { color: #f87171; }

    /* Export row */
    .mm-export-row { display:flex; gap:10px; flex-wrap:wrap; margin-top:14px; }
    </style>
    """, unsafe_allow_html=True)

    # ── Hero ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="mm-hero">
        <div class="mm-ai-badge">✦ AI Powered</div>
        <div class="mm-hero-title">AI Mind Map Generator</div>
        <div class="mm-hero-sub">
            Transform documents and ideas into interactive visual concept maps.
            Upload a file or type a prompt to generate your map instantly.
        </div>
        <div class="mm-chips">
            <div class="mm-chip">🧠 Smart Mapping</div>
            <div class="mm-chip">⚡ AI Powered</div>
            <div class="mm-chip">🌐 Interactive</div>
            <div class="mm-chip">📘 Concept Visualiser</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Two-column layout ────────────────────────────────────────────────────
    left_col, right_col = st.columns([1, 1.6], gap="large")

    # ── LEFT: inputs ─────────────────────────────────────────────────────────
    with left_col:

        # ── Map type selector ────────────────────────────────────────────────
        map_type = st.selectbox(
            "Map Type",
            ["🗺️  Mind Map", "🔗  Concept Map", "🌳  Topic Tree", "🕸️  Knowledge Graph"],
            label_visibility="collapsed",
        )

        # ════════════════════════════════════════
        # PANEL 1 — Document → Mind Map
        # ════════════════════════════════════════
        st.markdown("""
        <div class="mm-panel">
            <div class="mm-panel-title">📄 Document → Mind Map</div>
            <div class="mm-panel-sub">Upload a PDF, DOCX, or TXT and generate a map from its content.</div>
        </div>
        """, unsafe_allow_html=True)

        doc_file = st.file_uploader(
            "Upload document",
            type=["pdf", "docx", "txt"],
            label_visibility="collapsed",
            key="mm_doc_upload",
        )

        if doc_file:
            # Extract text (reuse existing loaders)
            from modules.document_loader import load_document
            doc_text = load_document(doc_file)
            st.caption(f"✅ {doc_file.name} loaded — {len(doc_text):,} characters")

            if st.button("🗺️ Generate Map from Document", type="primary",
                         use_container_width=True, key="mm_gen_doc"):
                with st.spinner("Analysing document with AI…"):
                    result = generate_mindmap(text=doc_text, source="document")
                if result["success"]:
                    st.session_state.mindmap_data = result["data"]
                    st.success("✅ Mind map generated!")
                else:
                    st.markdown(f"""
                    <div class="mm-error">
                        <strong>⚠️ Generation failed</strong><br>{result["error"]}
                    </div>
                    """, unsafe_allow_html=True)
                    st.session_state.mindmap_data = result["data"]  # show fallback

        st.markdown("---")

        # ════════════════════════════════════════
        # PANEL 2 — Chat → Mind Map
        # ════════════════════════════════════════
        st.markdown("""
        <div class="mm-panel">
            <div class="mm-panel-title">💬 Chat → Mind Map</div>
            <div class="mm-panel-sub">Type a topic or concept to visualise it instantly.</div>
        </div>
        """, unsafe_allow_html=True)

        # Chat history display
        if st.session_state.mindmap_chat:
            st.markdown('<div class="mm-chat-wrap">', unsafe_allow_html=True)
            for msg in st.session_state.mindmap_chat[-6:]:   # last 6 messages
                role_cls = "user" if msg["role"] == "user" else "ai"
                st.markdown(f"""
                <div class="mm-msg-row {role_cls}">
                    <div class="mm-bubble {role_cls}">{msg["content"]}</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Input
        with st.form("mm_chat_form", clear_on_submit=True):
            chat_cols = st.columns([4, 1])
            with chat_cols[0]:
                chat_input = st.text_input(
                    "Prompt",
                    placeholder="e.g. Generate a mind map for Machine Learning",
                    label_visibility="collapsed",
                )
            with chat_cols[1]:
                send_btn = st.form_submit_button("→", use_container_width=True)

        # Quick prompts
        quick_prompts = [
            "Machine Learning", "DBMS Concepts",
            "Python Basics", "Operating Systems",
        ]
        qp_cols = st.columns(2)
        for i, qp in enumerate(quick_prompts):
            if qp_cols[i % 2].button(f"💡 {qp}", use_container_width=True,
                                     key=f"qp_{i}"):
                chat_input = f"Generate a mind map for {qp}"
                send_btn = True

        if send_btn and chat_input.strip():
            st.session_state.mindmap_chat.append(
                {"role": "user", "content": chat_input}
            )
            with st.spinner("Generating mind map…"):
                result = generate_mindmap(prompt=chat_input, source="chat")

            if result["success"]:
                st.session_state.mindmap_data = result["data"]
                reply = f"✅ Mind map generated for **{result['data']['name']}** — see the visualisation →"
            else:
                reply = f"⚠️ {result['error']}"
                st.session_state.mindmap_data = result["data"]

            st.session_state.mindmap_chat.append(
                {"role": "ai", "content": reply}
            )
            st.rerun()

        # Clear chat
        if st.session_state.mindmap_chat:
            if st.button("🗑️ Clear Chat", use_container_width=True, key="mm_clear_chat"):
                st.session_state.mindmap_chat = []
                st.rerun()

    # ── RIGHT: visualisation ─────────────────────────────────────────────────
    with right_col:

        if st.session_state.mindmap_data:
            tree_json = json.dumps(st.session_state.mindmap_data, ensure_ascii=False)
            root_name = st.session_state.mindmap_data.get("name", "Mind Map")

            st.markdown(f"""
            <div class="mm-viz-wrap">
                <div class="mm-viz-toolbar">
                    <span class="mm-viz-title">🗺️ {root_name}</span>
                    <span class="mm-viz-hint">Scroll to zoom · Drag to pan · Hover nodes</span>
                </div>
            """, unsafe_allow_html=True)

            # ── D3.js radial tree ────────────────────────────────────────────
            d3_html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{
    background: transparent;
    font-family: 'DM Sans', 'Segoe UI', sans-serif;
    overflow: hidden;
  }}
  #mm-svg {{
    width: 100%;
    height: 560px;
    display: block;
  }}
  .link {{
    fill: none;
    stroke-width: 1.8;
    stroke-opacity: 0.55;
    transition: stroke-opacity 0.25s;
  }}
  .link:hover {{ stroke-opacity: 1; }}
  .node circle {{
    cursor: pointer;
    transition: r 0.2s ease, filter 0.2s ease;
  }}
  .node circle:hover {{
    filter: drop-shadow(0 0 10px currentColor);
  }}
  .node text {{
    font-size: 11px;
    fill: #e8edf5;
    pointer-events: none;
    font-weight: 500;
  }}
  .node.root circle {{
    filter: drop-shadow(0 0 14px #a78bfa);
  }}
  .node.root text {{
    font-size: 13px;
    font-weight: 700;
    fill: #e8edf5;
  }}
  .tooltip {{
    position: absolute;
    background: rgba(13,20,33,0.96);
    border: 1px solid rgba(99,102,241,0.35);
    border-radius: 10px;
    padding: 8px 13px;
    font-size: 12px;
    color: #e8edf5;
    pointer-events: none;
    opacity: 0;
    transition: opacity 0.2s;
    max-width: 200px;
    word-break: break-word;
    z-index: 99;
    backdrop-filter: blur(8px);
  }}
</style>
</head>
<body>
<svg id="mm-svg"></svg>
<div class="tooltip" id="tip"></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
<script>
(function() {{
  const DATA = {tree_json};

  const W = document.getElementById('mm-svg').clientWidth || 700;
  const H = 560;
  const R = Math.min(W, H) / 2 - 80;

  const svg = d3.select('#mm-svg')
    .attr('viewBox', [-W/2, -H/2, W, H])
    .call(d3.zoom()
      .scaleExtent([0.3, 3.5])
      .on('zoom', (e) => g.attr('transform', e.transform))
    );

  const g = svg.append('g');

  const tree = d3.tree()
    .size([2 * Math.PI, R])
    .separation((a, b) => (a.parent == b.parent ? 1 : 2) / a.depth);

  const root = d3.hierarchy(DATA);
  tree(root);

  /* ── Links ── */
  const linkGen = d3.linkRadial()
    .angle(d => d.x)
    .radius(d => d.y);

  function getColour(d) {{
    return d.data.color || '#6366f1';
  }}

  g.selectAll('.link')
    .data(root.links())
    .join('path')
      .attr('class', 'link')
      .attr('d', linkGen)
      .attr('stroke', d => getColour(d.target))
      .style('opacity', 0)
      .transition().duration(800).delay((d, i) => i * 18)
      .style('opacity', 1);

  /* ── Nodes ── */
  const node = g.selectAll('.node')
    .data(root.descendants())
    .join('g')
      .attr('class', d => 'node' + (d.depth === 0 ? ' root' : ''))
      .attr('transform', d =>
        `rotate(${{d.x * 180 / Math.PI - 90}}) translate(${{d.y}},0)`
      )
      .style('opacity', 0)
      .call(sel => sel.transition().duration(600)
        .delay((d, i) => i * 20)
        .style('opacity', 1)
      );

  node.append('circle')
    .attr('r', d => d.depth === 0 ? 14 : d.depth === 1 ? 9 : 6)
    .attr('fill', d => getColour(d))
    .attr('stroke', 'rgba(255,255,255,0.15)')
    .attr('stroke-width', 1.5)
    .on('mouseover', function(event, d) {{
      d3.select(this)
        .transition().duration(150)
        .attr('r', d.depth === 0 ? 18 : d.depth === 1 ? 12 : 8)
        .attr('filter', `drop-shadow(0 0 12px ${{getColour(d)}})`);
      const tip = document.getElementById('tip');
      tip.textContent = d.data.name;
      tip.style.opacity = 1;
      tip.style.left = (event.offsetX + 14) + 'px';
      tip.style.top  = (event.offsetY - 10) + 'px';
    }})
    .on('mousemove', function(event) {{
      const tip = document.getElementById('tip');
      tip.style.left = (event.offsetX + 14) + 'px';
      tip.style.top  = (event.offsetY - 10) + 'px';
    }})
    .on('mouseout', function(event, d) {{
      d3.select(this)
        .transition().duration(150)
        .attr('r', d.depth === 0 ? 14 : d.depth === 1 ? 9 : 6)
        .attr('filter', null);
      document.getElementById('tip').style.opacity = 0;
    }});

  node.append('text')
    .attr('dy', '0.31em')
    .attr('x', d => {{
      if (d.depth === 0) return 0;
      return (d.x < Math.PI) === !d.children ? 14 : -14;
    }})
    .attr('text-anchor', d => {{
      if (d.depth === 0) return 'middle';
      return (d.x < Math.PI) === !d.children ? 'start' : 'end';
    }})
    .attr('transform', d => {{
      if (d.depth === 0) return null;
      const angle = d.x * 180 / Math.PI - 90;
      return (d.x >= Math.PI) ? 'rotate(180)' : null;
    }})
    .text(d => {{
      const name = d.data.name;
      return name.length > 28 ? name.slice(0, 26) + '…' : name;
    }});

}})();
</script>
</body>
</html>
"""

            components.html(d3_html, height=562, scrolling=False)
            st.markdown("</div>", unsafe_allow_html=True)

            # ── Export controls ───────────────────────────────────────────────
            st.markdown('<div class="mm-export-row">', unsafe_allow_html=True)
            exp_c1, exp_c2, exp_c3 = st.columns(3)

            with exp_c1:
                json_bytes = json.dumps(
                    st.session_state.mindmap_data, indent=2, ensure_ascii=False
                ).encode()
                st.download_button(
                    "⬇️ Download JSON",
                    data=json_bytes,
                    file_name="mindmap.json",
                    mime="application/json",
                    use_container_width=True,
                )

            with exp_c2:
                if st.button("🔄 Regenerate", use_container_width=True, key="mm_regen"):
                    st.session_state.mindmap_data = None
                    st.rerun()

            with exp_c3:
                if st.button("🗑️ Clear Map", use_container_width=True, key="mm_clear"):
                    st.session_state.mindmap_data = None
                    st.session_state.mindmap_chat = []
                    st.rerun()

            st.markdown('</div>', unsafe_allow_html=True)

        else:
            # ── Empty state ───────────────────────────────────────────────────
            st.markdown("""
            <div style="
                display:flex; flex-direction:column; align-items:center;
                justify-content:center; height:480px;
                background: var(--surface);
                border: 2px dashed var(--border);
                border-radius: 20px;
                text-align:center;
                animation: fadeIn 0.5s ease;
            ">
                <div style="font-size:64px;margin-bottom:18px;
                    filter:drop-shadow(0 0 24px rgba(99,102,241,0.5));">🗺️</div>
                <div style="font-family:var(--font-head);font-size:20px;
                    font-weight:700;color:var(--text);margin-bottom:8px;">
                    Your Mind Map Appears Here
                </div>
                <div style="font-size:14px;color:var(--text2);max-width:320px;line-height:1.65;">
                    Upload a document or type a topic in the chat panel
                    to generate an interactive mind map.
                </div>
                <div style="display:flex;gap:12px;margin-top:24px;flex-wrap:wrap;
                    justify-content:center;">
                    <span style="padding:6px 14px;border-radius:99px;font-size:12px;
                        background:rgba(99,102,241,0.1);
                        border:1px solid rgba(99,102,241,0.25);color:#a78bfa;">
                        Zoom & Pan
                    </span>
                    <span style="padding:6px 14px;border-radius:99px;font-size:12px;
                        background:rgba(6,182,212,0.1);
                        border:1px solid rgba(6,182,212,0.25);color:#67e8f9;">
                        Hover Glow
                    </span>
                    <span style="padding:6px 14px;border-radius:99px;font-size:12px;
                        background:rgba(16,185,129,0.1);
                        border:1px solid rgba(16,185,129,0.25);color:#6ee7b7;">
                        Export JSON
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)


# When you paste into app.py, remove the outer triple-quotes and the
# MINDMAP_PAGE_CODE = ... wrapper — just paste the raw code block.
