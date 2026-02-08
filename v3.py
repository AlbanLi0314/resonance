import streamlit as st
import sys
import os
import pickle
import json
import time
import base64
from pathlib import Path

# Add project root to path so we can import src modules
sys.path.insert(0, str(Path(__file__).parent))

# Bridge Streamlit Cloud secrets to os.environ (for src.config compatibility)
if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
    os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]

from google import genai
from src.config import GEMINI_API_KEY, GEMINI_MODEL

# ============== Page Config ==============
st.set_page_config(
    page_title="Resonance",
    page_icon="✦",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ============== Load logo as base64 ==============
@st.cache_data
def get_logo_base64():
    logo_path = Path(__file__).parent / "assets" / "logo.jpg"
    if logo_path.exists():
        return base64.b64encode(logo_path.read_bytes()).decode()
    return None

logo_b64 = get_logo_base64()

# ============== Global CSS ==============
st.markdown("""
<style>
/* ====== Global Font & Background ====== */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    font-size: 18px !important;
}

/* Base font size */
p, li, span, div, label, .stMarkdown {
    font-size: 1.05rem !important;
}
h1 { font-size: 2.2rem !important; }
h2 { font-size: 1.8rem !important; }
h3 { font-size: 1.4rem !important; }

/* Main container gradient - dark blue/teal */
.stApp {
    background: linear-gradient(160deg, #080C10 0%, #0a1628 40%, #0c1a2a 100%);
}

/* ====== Hide Default Elements ====== */
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}
[data-testid="stSidebar"] {display: none;}
[data-testid="collapsedControl"] {display: none;}

/* ====== Glass Card ====== */
.glass-card {
    background: rgba(255, 255, 255, 0.04);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
    transition: all 0.3s ease;
}
.glass-card:hover {
    background: rgba(255, 255, 255, 0.07);
    border-color: rgba(0, 212, 255, 0.25);
    box-shadow: 0 8px 32px rgba(0, 212, 255, 0.08);
}

/* ====== Header Bar (Logo + Stepper) ====== */
.header-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 0 20px;
    margin-bottom: 8px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}
.logo-area {
    display: flex;
    align-items: center;
    gap: 14px;
}
.logo-area img {
    width: 48px;
    height: 48px;
    border-radius: 12px;
    object-fit: cover;
}
.logo-text {
    font-size: 1.6rem;
    font-weight: 700;
    letter-spacing: 4px;
    background: linear-gradient(135deg, #00D4FF 0%, #00F0FF 50%, #7DF9FF 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* ====== Stepper ====== */
.stepper {
    display: flex;
    align-items: center;
    gap: 0;
}
.step-item {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 16px;
    border-radius: 24px;
    font-size: 0.9rem;
    font-weight: 500;
    color: rgba(255,255,255,0.3);
    transition: all 0.3s ease;
    white-space: nowrap;
}
.step-item.active {
    background: rgba(0, 212, 255, 0.1);
    border: 1px solid rgba(0, 212, 255, 0.3);
    color: #00D4FF;
    font-weight: 600;
}
.step-item.done {
    color: rgba(0, 212, 255, 0.6);
}
.step-num {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 26px;
    height: 26px;
    border-radius: 50%;
    font-size: 0.8rem;
    font-weight: 700;
    background: rgba(255,255,255,0.06);
    color: rgba(255,255,255,0.3);
    flex-shrink: 0;
}
.step-item.active .step-num {
    background: rgba(0, 212, 255, 0.2);
    color: #00D4FF;
    box-shadow: 0 0 12px rgba(0, 212, 255, 0.3);
}
.step-item.done .step-num {
    background: rgba(0, 212, 255, 0.15);
    color: #00D4FF;
}
.step-connector {
    width: 32px;
    height: 2px;
    background: rgba(255,255,255,0.08);
    flex-shrink: 0;
}
.step-connector.done {
    background: rgba(0, 212, 255, 0.3);
}

/* ====== Upload Area ====== */
[data-testid="stFileUploader"] {
    border: 2px dashed rgba(0, 212, 255, 0.25) !important;
    border-radius: 16px !important;
    padding: 20px !important;
    background: rgba(0, 212, 255, 0.02) !important;
    transition: all 0.3s ease;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(0, 212, 255, 0.5) !important;
    background: rgba(0, 212, 255, 0.04) !important;
}

/* ====== Buttons ====== */
.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #00B4D8 0%, #00D4FF 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 32px !important;
    font-weight: 600 !important;
    color: #080C10 !important;
    letter-spacing: 0.3px;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 16px rgba(0, 212, 255, 0.25) !important;
}
.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="stBaseButton-primary"]:hover {
    box-shadow: 0 6px 24px rgba(0, 212, 255, 0.4) !important;
    transform: translateY(-1px);
}

.stButton > button[kind="secondary"],
.stButton > button[data-testid="stBaseButton-secondary"] {
    background: rgba(255, 255, 255, 0.06) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 12px !important;
    transition: all 0.3s ease !important;
}
.stButton > button[kind="secondary"]:hover,
.stButton > button[data-testid="stBaseButton-secondary"]:hover {
    background: rgba(255, 255, 255, 0.1) !important;
    border-color: rgba(0, 212, 255, 0.3) !important;
}

/* ====== Progress Bar ====== */
.stProgress > div > div {
    background: linear-gradient(90deg, #00B4D8, #00D4FF) !important;
    border-radius: 8px;
}
.stProgress > div {
    background: rgba(255, 255, 255, 0.06) !important;
    border-radius: 8px;
}

/* ====== Chat ====== */
[data-testid="stChatMessage"] {
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-radius: 16px !important;
    padding: 16px !important;
    margin-bottom: 8px;
}

[data-testid="stChatInput"] > div {
    border-radius: 16px !important;
    border: 1px solid rgba(0, 212, 255, 0.25) !important;
    background: rgba(255, 255, 255, 0.04) !important;
}

/* ====== Expander ====== */
[data-testid="stExpander"] {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-radius: 12px !important;
}

/* ====== Result Cards ====== */
.result-card {
    background: rgba(255, 255, 255, 0.04);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 20px;
    padding: 28px;
    margin-bottom: 20px;
    transition: all 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94);
    position: relative;
    overflow: hidden;
}
.result-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, #00B4D8, #00D4FF, #7DF9FF);
    opacity: 0;
    transition: opacity 0.3s ease;
}
.result-card:hover {
    background: rgba(255, 255, 255, 0.07);
    border-color: rgba(0, 212, 255, 0.2);
    box-shadow: 0 12px 40px rgba(0, 212, 255, 0.08);
    transform: translateY(-2px);
}
.result-card:hover::before {
    opacity: 1;
}

.result-rank {
    display: inline-block;
    background: linear-gradient(135deg, #00B4D8, #00D4FF);
    color: #080C10;
    font-size: 0.8rem;
    font-weight: 700;
    padding: 4px 14px;
    border-radius: 20px;
    margin-bottom: 12px;
}
.result-name {
    font-size: 1.7rem;
    font-weight: 700;
    color: #FAFAFA;
    margin-bottom: 4px;
}
.result-position {
    color: rgba(250, 250, 250, 0.5);
    font-size: 1.05rem;
    margin-bottom: 16px;
}
.result-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-bottom: 16px;
}
.meta-tag {
    background: rgba(0, 212, 255, 0.08);
    border: 1px solid rgba(0, 212, 255, 0.18);
    color: #7DF9FF;
    font-size: 0.95rem;
    padding: 6px 14px;
    border-radius: 20px;
}
.result-reason {
    background: rgba(255, 255, 255, 0.03);
    border-left: 3px solid #00D4FF;
    padding: 16px 20px;
    border-radius: 0 8px 8px 0;
    color: rgba(250, 250, 250, 0.75);
    font-size: 1.05rem;
    line-height: 1.7;
    margin: 16px 0;
}

/* Score badge */
.score-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(0, 212, 255, 0.1);
    border: 1px solid rgba(0, 212, 255, 0.25);
    padding: 8px 16px;
    border-radius: 20px;
    font-size: 1rem;
    font-weight: 600;
    color: #7DF9FF;
    float: right;
}
.score-high { color: #34d399; border-color: rgba(52, 211, 153, 0.3); background: rgba(52, 211, 153, 0.1); }
.score-mid { color: #60a5fa; border-color: rgba(96, 165, 250, 0.3); background: rgba(96, 165, 250, 0.1); }
.score-low { color: #fbbf24; border-color: rgba(251, 191, 36, 0.3); background: rgba(251, 191, 36, 0.1); }

/* ====== Divider ====== */
hr {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0, 212, 255, 0.15), transparent);
    margin: 24px 0;
}

/* ====== Alert boxes ====== */
[data-testid="stAlert"] {
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
}

/* ====== Spinner ====== */
.stSpinner > div {
    border-top-color: #00D4FF !important;
}

/* ====== Contact Info ====== */
.contact-bar {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    margin-top: 12px;
}
.contact-item {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 10px 18px;
    font-size: 1rem;
    color: #e0e0e0;
    transition: all 0.3s ease;
}
.contact-item:hover {
    background: rgba(0, 212, 255, 0.08);
    border-color: rgba(0, 212, 255, 0.25);
}
.contact-item .contact-icon {
    font-size: 1.1rem;
}
.contact-item a {
    color: #7DF9FF;
    text-decoration: none;
}
.contact-item a:hover {
    color: #00F0FF;
    text-decoration: underline;
}
.contact-email {
    font-family: 'Inter', monospace;
    color: #e0e0e0;
    user-select: all;
}

/* ====== Footer ====== */
.footer-text {
    text-align: center;
    color: rgba(250, 250, 250, 0.2);
    font-size: 0.85rem;
    padding: 32px 0 16px;
}

/* ====== Animations ====== */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}
.animate-in {
    animation: fadeInUp 0.5s ease-out forwards;
}

/* ====== Section title ====== */
.section-title {
    font-size: 1.5rem;
    font-weight: 600;
    color: #FAFAFA;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-title .icon {
    font-size: 1.3rem;
}
</style>
""", unsafe_allow_html=True)


# ============== Model Cache (loaded once) ==============

@st.cache_resource
def load_search_engine():
    """Load SPECTER2 encoder + researcher data + build HybridSearch"""
    from src.embedding import Specter2Encoder
    from src.hybrid_search import HybridSearch

    pkl_path = Path(__file__).parent / "data" / "processed" / "researchers_dual_adapter.pkl"
    with open(pkl_path, "rb") as f:
        researchers = pickle.load(f)

    encoder = Specter2Encoder()
    encoder.load()
    hybrid_search = HybridSearch(researchers, encoder)
    return hybrid_search, researchers


@st.cache_resource
def load_gemini():
    """Return Gemini client"""
    return genai.Client(api_key=GEMINI_API_KEY)


# ============== Preload Models (with splash screen) ==============
_splash = st.empty()
_splash.markdown(f"""
<div id="splash-screen" style="
    position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
    background: radial-gradient(ellipse at center, #0a1628 0%, #080C10 70%);
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    z-index: 999999;
">
    <!-- Animated rings -->
    <div style="position: relative; width: 160px; height: 160px; margin-bottom: 32px;">
        <div style="
            position: absolute; inset: 0;
            border: 2px solid rgba(0, 212, 255, 0.15);
            border-radius: 50%;
            animation: pulse-ring 2s ease-out infinite;
        "></div>
        <div style="
            position: absolute; inset: 20px;
            border: 2px solid rgba(0, 212, 255, 0.25);
            border-radius: 50%;
            animation: pulse-ring 2s ease-out 0.4s infinite;
        "></div>
        <div style="
            position: absolute; inset: 40px;
            border: 2px solid rgba(0, 212, 255, 0.4);
            border-radius: 50%;
            animation: pulse-ring 2s ease-out 0.8s infinite;
        "></div>
        <!-- Center glow -->
        <div style="
            position: absolute; inset: 55px;
            background: radial-gradient(circle, rgba(0, 212, 255, 0.4) 0%, transparent 70%);
            border-radius: 50%;
            animation: glow-pulse 1.5s ease-in-out infinite alternate;
        "></div>
        <!-- Cross lines -->
        <div style="
            position: absolute; top: 50%; left: 10%; right: 10%; height: 1px;
            background: linear-gradient(90deg, transparent, rgba(0, 212, 255, 0.5), transparent);
            animation: glow-pulse 1.5s ease-in-out infinite alternate;
        "></div>
        <div style="
            position: absolute; left: 50%; top: 10%; bottom: 10%; width: 1px;
            background: linear-gradient(180deg, transparent, rgba(0, 212, 255, 0.5), transparent);
            animation: glow-pulse 1.5s ease-in-out infinite alternate;
        "></div>
    </div>
    <!-- Brand name -->
    <div style="
        font-family: 'Inter', sans-serif;
        font-size: 2.2rem; font-weight: 700; letter-spacing: 8px;
        background: linear-gradient(135deg, #00D4FF 0%, #00F0FF 50%, #7DF9FF 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
        margin-bottom: 16px;
    ">RESONANCE</div>
    <!-- Loading text -->
    <div style="
        color: rgba(0, 212, 255, 0.5); font-size: 0.95rem; letter-spacing: 2px;
        animation: loading-dots 1.5s ease-in-out infinite;
    ">Initializing AI Models</div>
</div>

<style>
@keyframes pulse-ring {{
    0% {{ transform: scale(0.95); opacity: 1; }}
    100% {{ transform: scale(1.15); opacity: 0; }}
}}
@keyframes glow-pulse {{
    0% {{ opacity: 0.4; }}
    100% {{ opacity: 1; }}
}}
@keyframes loading-dots {{
    0%, 100% {{ opacity: 0.4; }}
    50% {{ opacity: 1; }}
}}
</style>
""", unsafe_allow_html=True)

import time as _time
_t0 = _time.time()
hybrid_search, researchers = load_search_engine()
gemini_client = load_gemini()
_elapsed = _time.time() - _t0
if _elapsed < 2.5:
    _time.sleep(2.5 - _elapsed)
_splash.empty()


# ============== Backend Functions ==============

def extract_pdf_text(uploaded_file) -> str:
    from PyPDF2 import PdfReader
    import io
    pdf_bytes = uploaded_file.getvalue()
    reader = PdfReader(io.BytesIO(pdf_bytes))
    texts = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            texts.append(text)
    return "\n".join(texts)


def analyze_paper(paper_text: str) -> dict:
    truncated = paper_text[:8000]
    prompt = f"""Analyze this academic paper and extract key information.

Paper text:
{truncated}

Return a JSON object with:
{{
    "title": "the paper's title",
    "abstract": "a 2-3 sentence summary of the paper",
    "topics": ["topic1", "topic2", "topic3", "topic4", "topic5"]
}}

The topics should be specific research areas/techniques covered in this paper (3-5 topics).
Output ONLY the JSON, nothing else."""

    try:
        response = gemini_client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        result_text = response.text.strip()
        if "```json" in result_text:
            json_start = result_text.find("```json") + 7
            json_end = result_text.find("```", json_start)
            result_text = result_text[json_start:json_end].strip()
        elif "{" in result_text:
            json_start = result_text.find("{")
            json_end = result_text.rfind("}") + 1
            result_text = result_text[json_start:json_end]
        return json.loads(result_text)
    except Exception as e:
        return {"title": "Unable to parse", "abstract": f"Analysis failed: {str(e)[:100]}", "topics": ["Unknown"]}


def chat_about_paper(paper_text: str, history: list, user_message: str = None) -> str:
    truncated_paper = paper_text[:6000]
    system_instruction = f"""You are an academic research assistant helping a user find researchers at Cornell University to discuss a paper with.

The user uploaded this paper:
{truncated_paper}

Your job:
1. First, analyze the paper and ask the user what aspects they're most interested in discussing
2. Through conversation, understand exactly what direction/topic they want to explore
3. When you feel confident you understand their need (usually after 1-2 exchanges), confirm with them
4. Once confirmed, append this EXACT tag at the end of your message: [SEARCH_READY: <a concise English search query for finding matching researchers>]

Important rules:
- Respond in English
- Be conversational and friendly
- The search query in [SEARCH_READY: ...] must be in English and be a good academic search query
- Do NOT output the [SEARCH_READY: ...] tag until the user has confirmed their interest
- Keep responses concise (under 200 words)"""

    messages = []
    if user_message is None:
        messages.append({"role": "user", "parts": [{"text": "Please analyze this paper and ask me which aspects I'd like to discuss."}]})
    else:
        for msg in history:
            role = "user" if msg["role"] == "user" else "model"
            messages.append({"role": role, "parts": [{"text": msg["content"]}]})
        messages.append({"role": "user", "parts": [{"text": user_message}]})

    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=messages,
            config=genai.types.GenerateContentConfig(system_instruction=system_instruction)
        )
        return response.text.strip()
    except Exception as e:
        return f"AI error: {str(e)[:100]}"


def search_researchers(query: str, top_k: int = 5) -> list:
    from src.query_expansion import expand_query
    from src.match_explainer import explain_matches_batch

    try:
        expanded_query = expand_query(query)
    except Exception:
        expanded_query = query

    raw_results = hybrid_search.search(expanded_query, top_k=top_k)

    try:
        explained = explain_matches_batch(query, raw_results, top_k=top_k)
    except Exception:
        explained = raw_results

    results = []
    for i, r in enumerate(explained):
        results.append({
            "name": r.get("name", "Unknown"),
            "name_cn": None,
            "position": r.get("position", ""),
            "department": r.get("department", ""),
            "lab": r.get("lab", ""),
            "advisor": None,
            "email": r.get("email", ""),
            "website": r.get("personal_website"),
            "match_score": int(r.get("embedding_score", 0) * 100),
            "reason": r.get("explanation", r.get("reason", "Research area match")),
            "papers": [],
            "key_overlap": r.get("key_overlap", []),
            "match_type": r.get("match_type", ""),
            "confidence": r.get("confidence", 50),
        })
    return results


# ============== Helper: Render Result Card ==============

def render_result_card(i: int, person: dict):
    score = person['match_score']
    score_class = "score-high" if score >= 90 else "score-mid" if score >= 75 else "score-low"

    meta_html = ""
    if person.get('department'):
        meta_html += f'<span class="meta-tag">{person["department"]}</span>'
    if person.get('lab'):
        meta_html += f'<span class="meta-tag">{person["lab"]}</span>'
    if person.get('key_overlap'):
        for kw in person['key_overlap'][:3]:
            meta_html += f'<span class="meta-tag">{kw}</span>'

    contact_html = '<div class="contact-bar">'
    if person.get('email'):
        contact_html += f'<div class="contact-item"><span class="contact-icon">✉️</span> <span class="contact-email">{person["email"]}</span></div>'
    if person.get('website'):
        contact_html += f'<div class="contact-item"><span class="contact-icon">🔗</span> <a href="{person["website"]}" target="_blank">{person["website"]}</a></div>'
    contact_html += '</div>'

    st.markdown(f"""
    <div class="result-card animate-in" style="animation-delay: {i * 0.1}s">
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
            <div>
                <span class="result-rank">#{i+1}</span>
                <div class="result-name">{person['name']}</div>
                <div class="result-position">{person['position']}</div>
            </div>
            <span class="score-badge {score_class}">{score}% match</span>
        </div>
        <div class="result-meta">{meta_html}</div>
        <div class="result-reason">{person['reason']}</div>
        {contact_html}
    </div>
    """, unsafe_allow_html=True)


# ============== Init Session State ==============
for key, default in {
    "step": 1,
    "chat_history": [],
    "paper_analyzed": False,
    "user_need_confirmed": False,
    "paper_text": "",
    "final_query": "",
    "search_results": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def reset_all():
    st.session_state.step = 1
    st.session_state.chat_history = []
    st.session_state.paper_analyzed = False
    st.session_state.user_need_confirmed = False
    st.session_state.paper_text = ""
    st.session_state.final_query = ""
    st.session_state.search_results = []


def reset_chat():
    st.session_state.step = 2
    st.session_state.chat_history = []
    st.session_state.paper_analyzed = False
    st.session_state.user_need_confirmed = False
    st.session_state.final_query = ""
    st.session_state.search_results = []


# ============== Header Bar: Logo + Stepper ==============
current = st.session_state.step
step_labels = ["Upload", "Discuss", "Match"]

# Build stepper HTML
stepper_html = ""
for i, label in enumerate(step_labels, 1):
    cls = "done" if i < current else "active" if i == current else ""
    check = "✓" if i < current else str(i)
    stepper_html += f'<div class="step-item {cls}"><span class="step-num">{check}</span> {label}</div>'
    if i < 3:
        conn_cls = "done" if i < current else ""
        stepper_html += f'<div class="step-connector {conn_cls}"></div>'

logo_img = f'<img src="data:image/jpeg;base64,{logo_b64}" />' if logo_b64 else ''

st.markdown(f"""
<div class="header-bar animate-in">
    <div class="logo-area">
        {logo_img}
        <span class="logo-text">RESONANCE</span>
    </div>
    <div class="stepper">
        {stepper_html}
    </div>
</div>
""", unsafe_allow_html=True)


# ============== Step 1: Upload Paper ==============
if st.session_state.step == 1:
    st.markdown("""
    <div class="glass-card animate-in" style="text-align: center; padding: 40px 24px;">
        <div style="font-size: 3rem; margin-bottom: 12px;">📄</div>
        <div style="font-size: 1.4rem; font-weight: 600; margin-bottom: 8px;">Upload Your Paper</div>
        <div style="color: rgba(250,250,250,0.45); font-size: 1.05rem;">
            Drop a PDF of the paper you'd like to discuss with someone at Cornell
        </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose PDF",
        type=["pdf"],
        help="Text-based PDFs work best (not scanned images)",
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        file_size = len(uploaded_file.getvalue()) / 1024
        st.markdown(f"""
        <div class="glass-card" style="padding: 16px 24px;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <span style="font-size: 1.5rem;">✅</span>
                <div>
                    <div style="font-weight: 600; font-size: 1.05rem;">{uploaded_file.name}</div>
                    <div style="color: rgba(250,250,250,0.4); font-size: 0.9rem;">{file_size:.0f} KB</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("Analyze Paper", use_container_width=True, type="primary"):
                with st.spinner("Extracting text..."):
                    paper_text = extract_pdf_text(uploaded_file)
                if not paper_text.strip():
                    st.error("Could not extract text. Make sure this isn't a scanned PDF.")
                else:
                    with st.spinner("AI is analyzing your paper..."):
                        paper_info = analyze_paper(paper_text)
                    st.session_state.uploaded_file = uploaded_file
                    st.session_state.paper_text = paper_text
                    st.session_state.paper_info = paper_info
                    st.session_state.step = 2
                    st.rerun()
    else:
        st.markdown("""
        <div style="text-align: center; color: rgba(250,250,250,0.3); padding: 16px; font-size: 1rem;">
            Drag & drop or click to browse
        </div>
        """, unsafe_allow_html=True)


# ============== Step 2: AI Conversation ==============
elif st.session_state.step == 2:

    with st.expander("📄 Paper Details", expanded=False):
        st.markdown(f"**{st.session_state.paper_info['title']}**")
        st.caption(st.session_state.paper_info['abstract'][:300])
        if st.session_state.paper_info.get('topics'):
            topics_str = " &ensp;|&ensp; ".join(st.session_state.paper_info['topics'])
            st.markdown(f"*{topics_str}*")

    if not st.session_state.paper_analyzed:
        with st.spinner("AI is reading your paper..."):
            initial_response = chat_about_paper(st.session_state.paper_text, [])
        st.session_state.chat_history.append({"role": "assistant", "content": initial_response})
        st.session_state.paper_analyzed = True

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            display_text = message["content"]
            if "[SEARCH_READY:" in display_text:
                display_text = display_text[:display_text.index("[SEARCH_READY:")].strip()
            st.markdown(display_text)

    if st.session_state.user_need_confirmed:
        st.markdown(f"""
        <div class="glass-card" style="text-align: center; border-color: rgba(0, 212, 255, 0.3);">
            <div style="color: #00D4FF; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;">Search Query Ready</div>
            <div style="font-size: 1.15rem; font-weight: 600;">{st.session_state.final_query}</div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("Find Researchers", use_container_width=True, type="primary"):
                with st.spinner(f"Searching across {len(researchers)} researchers..."):
                    results = search_researchers(st.session_state.final_query)
                    st.session_state.search_results = results
                st.session_state.step = 3
                st.rerun()
    else:
        user_input = st.chat_input("Tell me what interests you about this paper...")
        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with st.spinner("Thinking..."):
                ai_response = chat_about_paper(
                    st.session_state.paper_text,
                    st.session_state.chat_history[:-1],
                    user_input
                )
            if "[SEARCH_READY:" in ai_response:
                tag_start = ai_response.index("[SEARCH_READY:") + len("[SEARCH_READY:")
                tag_end = ai_response.index("]", tag_start)
                st.session_state.final_query = ai_response[tag_start:tag_end].strip()
                st.session_state.user_need_confirmed = True
            st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
            st.rerun()


# ============== Step 3: Results ==============
elif st.session_state.step == 3:
    results = st.session_state.search_results

    st.markdown(f"""
    <div class="glass-card" style="text-align: center; margin-bottom: 28px;">
        <div style="color: rgba(250,250,250,0.45); font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px;">Matches Found</div>
        <div style="font-size: 2.4rem; font-weight: 700; background: linear-gradient(135deg, #00B4D8, #00D4FF); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">{len(results)}</div>
        <div style="color: rgba(250,250,250,0.5); font-size: 1rem; margin-top: 4px;">for &ldquo;{st.session_state.final_query}&rdquo;</div>
    </div>
    """, unsafe_allow_html=True)

    for i, person in enumerate(results):
        render_result_card(i, person)
        if i < len(results) - 1:
            st.markdown("<div style='height: 8px'></div>", unsafe_allow_html=True)

    st.markdown("<div style='height: 24px'></div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("Refine Search", use_container_width=True):
            reset_chat()
            st.rerun()
    with col3:
        if st.button("New Paper", use_container_width=True):
            reset_all()
            st.rerun()


# ============== Footer ==============
st.markdown("""
<div class="footer-text">
    Powered by Gemini 3 &ensp;|&ensp; Built for the Google DeepMind Gemini 3 Hackathon
</div>
""", unsafe_allow_html=True)
