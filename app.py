import streamlit as st
import os

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
# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Academic Assistant", layout="wide")

# ---------------- SESSION STATE ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------- PREMIUM UI ----------------
st.markdown("""
<style>
.stApp {
    background-color: #0f172a;
    color: #e2e8f0;
}

h1 {
    text-align: center;
    color: #60a5fa;
}

/* Chat bubbles */
.user-msg {
    background: #2563eb;
    padding: 12px;
    border-radius: 12px;
    margin: 6px 0;
}

.bot-msg {
    background: #1f2937;
    padding: 12px;
    border-radius: 12px;
    margin: 6px 0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #111827;
}

/* Cards */
.card {
    background: #1e293b;
    padding: 12px;
    border-radius: 10px;
    margin: 8px 0;
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("🎓 AI Academic Assistant ")

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.markdown("## 📂 Document Manager")

files = st.sidebar.file_uploader("Upload Files", accept_multiple_files=True)
url = st.sidebar.text_input("Enter URL")

# PROCESS DOCUMENTS
if st.sidebar.button("🚀 Process Documents"):
    text_data = ""

    if files:
        for file in files:
            text_data += load_document(file)

    if url:
        text_data += load_url(url)

    if text_data.strip():
        with st.spinner("Processing documents..."):
            chunks = split_text(text_data)
            chunks = chunks[:20]  # SPEED OPTIMIZATION

            embeddings = create_embeddings(chunks)
            save_index(embeddings, chunks)

        st.sidebar.success("✅ Ready for Chat!")
    else:
        st.sidebar.warning("No input provided")

# CLEAR INDEX
if st.sidebar.button("🗑️ Clear Index"):
    if os.path.exists("data/faiss_index/index.faiss"):
        os.remove("data/faiss_index/index.faiss")
    if os.path.exists("data/faiss_index/texts.pkl"):
        os.remove("data/faiss_index/texts.pkl")

    st.sidebar.success("Cleared!")

# CHAT HISTORY
st.sidebar.markdown("---")
st.sidebar.markdown("## 💬 Chat History")

for i, chat in enumerate(st.session_state.chat_history[-10:]):
    st.sidebar.markdown(f"**Q{i+1}:** {chat['q']}")

# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3 = st.tabs([
    "💬 Chat",
    "📝 Questions",
    "📄 Analyzer"
])

# =========================================================
# 🎤 VOICE INPUT (GLOBAL - FIXED)
# =========================================================
voice_text = ""

with st.sidebar:
    st.markdown("## 🎤 Voice Input")

    audio_file = st.file_uploader("Upload Voice (wav/mp3)", type=["wav", "mp3"])

    if audio_file is not None:
        from modules.audio_loader import transcribe_audio

        with st.spinner("Transcribing voice... 🎧"):
            voice_text = transcribe_audio(audio_file)

        st.success("Voice converted ✅")
        st.write(voice_text)

# =========================================================
# 💬 CHAT TAB
# =========================================================
with tab1:
    query = st.chat_input("Ask anything from your documents...")

    # 🎯 Voice overrides text input (ChatGPT-style behavior)
    if voice_text:
        query = voice_text

    if query:
        index, texts = load_index()

        if index is None:
            st.warning("⚠️ Please process documents first")
            st.stop()

        # save chat history
        st.session_state.chat_history.append({"q": query})

        with st.spinner("AI is thinking... 🤖"):
            answer, chunks = generate_answer(query, index, texts)

        # USER MESSAGE
        st.markdown(f"""
        <div class="user-msg">
        🧑‍🎓 <b>You:</b><br>{query}
        </div>
        """, unsafe_allow_html=True)

        # BOT MESSAGE
        st.markdown(f"""
        <div class="bot-msg">
        🤖 <b>Assistant:</b><br>{answer}
        </div>
        """, unsafe_allow_html=True)

        # SOURCES
        with st.expander("📚 Source Context"):
            for c in chunks:
                st.markdown(f"<div class='card'>{c}</div>", unsafe_allow_html=True)

# =========================================================
# 📝 QUESTION GENERATOR
# =========================================================
with tab2:
    st.subheader("📝 Question Paper Generator")

    difficulty = st.selectbox("Select Difficulty", ["Easy", "Medium", "Hard"])

    if st.button("Generate"):
        index, texts = load_index()

        if index is None:
            st.warning("Process documents first")
            st.stop()

        text = " ".join(texts)

        with st.spinner("Generating questions..."):
            result = generate_questions(text, difficulty)

        st.markdown(f"<div class='card'>{result}</div>", unsafe_allow_html=True)

# =========================================================
# 📄 ANALYZER
# =========================================================
with tab3:
    st.subheader("📄 Research Paper Analyzer")

    if st.button("Analyze"):
        index, texts = load_index()

        if index is None:
            st.warning("Process documents first")
            st.stop()

        text = " ".join(texts)

        with st.spinner("Analyzing paper..."):
            result = analyze_paper(text)

        st.markdown(f"<div class='card'>{result}</div>", unsafe_allow_html=True)