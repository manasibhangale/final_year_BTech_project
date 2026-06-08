import streamlit as st
import os
import time

from modules.document_loader import load_document, load_url
from modules.text_splitter import split_text
from modules.vector_store import create_embeddings, save_index, load_index
from modules.rag_pipeline import generate_answer
from modules.question_generator import generate_questions
from modules.research_analyzer import analyze_paper
import warnings
import logging

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

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