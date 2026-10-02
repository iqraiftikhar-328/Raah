import os, re, glob
import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & CUSTOM CSS
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Raah", page_icon="🧭", layout="wide")

st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* ================================================================= */
    /* THE NUCLEAR OPTION: Force Light Mode & Plum Text Everywhere       */
    /* ================================================================= */
    html, body, [class*="css"], .stApp, .stMarkdown, p, h1, h2, h3, h4, h5, h6, label, span, div {
        font-family: 'Inter', sans-serif;
        color: #50394c !important; /* Force all text to Deep Plum */
    }
    
    /* Force App Background to Soft Peach */
    .stApp {
        background-color: #f4e1d2 !important;
    }

    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* --- HERO SECTION --- */
    .hero {
        background: linear-gradient(135deg, #50394c 0%, #b2b2b2 100%);
        padding: 3rem 2rem;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 15px -3px rgba(80, 57, 76, 0.3);
    }
    .hero-badge {
        background-color: #ffef96;
        color: #50394c !important;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        display: inline-block;
        margin-bottom: 1rem;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .hero h1 {
        font-size: 2.8rem;
        font-weight: 700;
        margin: 0 0 1rem 0;
        color: #ffffff !important; /* Force white for contrast */
    }
    .hero p {
        font-size: 1.1rem;
        max-width: 600px;
        margin: 0 auto;
        opacity: 0.95;
        line-height: 1.6;
        color: #f4e1d2 !important;
    }

    /* --- WARNING BANNER --- */
    .warn {
        background: #ffef96;
        border-left: 6px solid #50394c;
        color: #50394c !important;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        font-size: 0.9rem;
        font-weight: 500;
        margin-bottom: 2rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .warn * { color: #50394c !important; }

    /* --- SIDEBAR (Force Dark Plum Background & Peach Text) --- */
    [data-testid="stSidebar"] {
        background-color: #50394c !important;
        border-right: 1px solid #b2b2b2;
        padding-top: 2rem;
    }
    /* Force all sidebar text to be light peach */
    [data-testid="stSidebar"] *, [data-testid="stSidebar"] p, [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] label, [data-testid="stSidebar"] span {
        color: #f4e1d2 !important;
    }
    /* Force Dropdowns to be Peach with Plum text */
    [data-testid="stSidebar"] div[data-baseweb="select"] {
        background-color: #f4e1d2 !important;
        border-radius: 8px;
    }
    [data-testid="stSidebar"] div[data-baseweb="select"] * {
        color: #50394c !important;
        background-color: transparent !important;
    }
    
    .logo-area {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 2rem;
    }
    .logo-icon {
        background-color: #ffef96;
        color: #50394c !important;
        width: 45px;
        height: 45px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
    .logo-text h2 { margin: 0; font-size: 1.4rem; color: #ffef96 !important; }
    .logo-text p { margin: 0; font-size: 0.8rem; color: #b2b2b2 !important; }

    /* --- MAIN INPUTS (Force White Background) --- */
    .stTextInput > div > div > input, .stNumberInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #b2b2b2;
        padding: 14px 18px;
        font-size: 1rem;
        background-color: #ffffff !important;
        color: #50394c !important;
        transition: all 0.3s ease;
    }
    .stTextInput > div > div > input::placeholder, .stNumberInput > div > div > input::placeholder {
        color: #b2b2b2 !important;
    }
    .stTextInput > div > div > input:focus, .stNumberInput > div > div > input:focus {
        border-color: #50394c;
        box-shadow: 0 0 0 3px rgba(80, 57, 76, 0.2);
    }

    /* --- BUTTONS --- */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        padding: 10px 20px;
        background-color: #ffffff !important;
        color: #50394c !important;
        border: 1px solid #b2b2b2;
        transition: all 0.2s ease;
        width: 100%;
    }
    .stButton > button:hover {
        background-color: #ffef96 !important;
        border-color: #50394c;
        color: #50394c !important;
    }
    /* Force text inside buttons to be visible */
    .stButton > button * { color: inherit !important; }

    .stButton > button[kind="primary"] {
        background-color: #50394c !important;
        color: #f4e1d2 !important;
        border: none;
        padding: 12px 30px;
        font-size: 1.05rem;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #ffef96 !important;
        color: #50394c !important;
        box-shadow: 0 6px 12px rgba(80, 57, 76, 0.3);
        transform: translateY(-1px);
    }

    /* --- TABS --- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        background-color: transparent;
        border-bottom: none !important;
        margin-bottom: 2rem;
        padding-bottom: 0px;
    }
    .stTabs [data-baseweb="tab-highlight"] { display: none !important; }
    .stTabs [data-baseweb="tab-border"] { display: none !important; }
    
    .stTabs [data-baseweb="tab"] {
        height: auto;
        padding: 14px 28px;
        background-color: #ffffff !important;
        border-radius: 10px;
        border: 2px solid #b2b2b2;
        color: #50394c !important;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.2s ease;
        margin: 0;
    }
    .stTabs [data-baseweb="tab"] * { color: #50394c !important; }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #ffef96 !important;
        border-color: #50394c;
    }
    .stTabs [aria-selected="true"] {
        background-color: #50394c !important;
        border-color: #50394c !important;
        box-shadow: 0 4px 6px rgba(80, 57, 76, 0.3);
    }
    .stTabs [aria-selected="true"] * { color: #f4e1d2 !important; }

    /* --- SOURCE CARDS --- */
    .src {
        border-left: 5px solid #50394c;
        background: #ffffff !important;
        padding: 14px 18px;
        border-radius: 8px;
        margin: 10px 0;
        color: #50394c !important;
        font-size: 0.9rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        border: 1px solid #b2b2b2;
    }
    .src * { color: #50394c !important; }
    .src a { color: #1d4ed8 !important; text-decoration: underline; font-weight: 600;}

    /* --- SHORTLIST CARDS --- */
    .uni-card {
        background: #ffffff !important;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #b2b2b2;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
        border-top: 5px solid #50394c;
    }
    .uni-card * { color: #50394c !important; }
    
    .uni-tag {
        display: inline-block;
        background: #f4e1d2 !important;
        color: #50394c !important;
        padding: 6px 12px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 8px;
        margin-bottom: 8px;
        border: 1px solid #b2b2b2;
    }
    
    /* --- EMPTY STATE --- */
    .empty-state {
        text-align: center; 
        padding: 4rem 2rem; 
        background: #ffffff !important; 
        border-radius: 12px; 
        border: 2px dashed #b2b2b2; 
        margin-top: 1rem;
    }
    .empty-state * { color: #50394c !important; }
    .empty-state p { color: #b2b2b2 !important; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. BACKEND LOGIC
# -----------------------------------------------------------------------------
def secret(name, default=None):
    try:
        return st.secrets[name]
    except Exception:
        return os.getenv(name, default)

MODEL = secret("GROQ_MODEL", "openai/gpt-oss-120b")
MIN_SCORE = 0.25
NOT_FOUND = "I could not find this in the available knowledge base. Please check the official university website."
SYSTEM = ("You are Raah, a career and admissions assistant. Answer ONLY using the provided context. "
          "If the answer is not in the context, say it was not found in the available knowledge base. "
          "Never invent admission rules, fees, dates, salaries or job prospects. Mention uncertainty. Be concise and clear.")

def load_chunks():
    chunks = []
    for path in sorted(glob.glob("data/*.txt")):
        if "sample_template" in path:
            continue
        text = open(path, encoding="utf-8").read()
        head, _, body = text.partition("\n\n")
        meta = {}
        for line in head.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip().lower()] = v.strip()
        for block in re.split(r"\n(?=## )", body):
            lines = block.strip().splitlines()
            if len(lines) < 2:
                continue
            section = lines[0].replace("##", "").strip()
            content = " ".join(lines[1:]).strip()
            if len(content) < 20:
                continue
            chunks.append({"text": f"{meta.get('university','')} - {meta.get('field','')} - {section}: {content}",
                           "university": meta.get("university", "?"), "city": meta.get("city", "?"),
                           "field": meta.get("field", "?"), "year": meta.get("year", "?"),
                           "source": meta.get("source", ""), "section": section})
    return chunks

@st.cache_resource(show_spinner="Building knowledge base (one time)...")
def build_index():
    chunks = load_chunks()
    if not chunks:
        return None, None, []
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    emb = model.encode([c["text"] for c in chunks], normalize_embeddings=True).astype("float32")
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    return model, index, chunks

model, index, chunks = build_index()

def retrieve(query, field="All", city="All", k=3):
    if not model or not index or not chunks:
        return []
    qv = model.encode([query], normalize_embeddings=True).astype("float32")
    scores, ids = index.search(qv, len(chunks))
    hits = []
    for s, i in zip(scores[0], ids[0]):
        c = chunks[i]
        if s < MIN_SCORE or (field != "All" and c["field"] != field) or (city != "All" and c["city"] != city):
            continue
        hits.append({**c, "score": float(s)})
    return hits[:k]

def llm(user_msg):
    client = Groq(api_key=secret("GROQ_API_KEY"))
    r = client.chat.completions.create(model=MODEL, temperature=0.1,
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_msg}])
    return r.choices[0].message.content

def show_sources(hits):
    with st.expander("📚 View Sources Used"):
        for h in hits:
            st.markdown(f'<div class="src"><b>{h["university"]}</b> — {h["section"]} ({h["field"]}, {h["city"]}, {h["year"]})<br>'
                        f'<a href="{h["source"]}" target="_blank">{h["source"]}</a> &nbsp;|&nbsp; Match Score: {h["score"]:.2f}</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. ERROR HANDLING
# -----------------------------------------------------------------------------
if not chunks:
    st.error("No documents found. Add .txt files to the data/ folder (see data/sample_template.txt).")
    st.stop()

# -----------------------------------------------------------------------------
# 4. SIDEBAR
# -----------------------------------------------------------------------------
fields = ["All"] + sorted({c["field"] for c in chunks})
cities = ["All"] + sorted({c["city"] for c in chunks})

with st.sidebar:
    st.markdown("""
    <div class="logo-area">
        <div class="logo-icon">🧭</div>
        <div class="logo-text">
            <h2>Raah</h2>
            <p>Career Navigator</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Refine your search")
    f_field = st.selectbox("Field of Study", fields)
    f_city = st.selectbox("City", cities)
    
    st.markdown("---")
    st.caption(f"📊 **{len(chunks)}** chunks from **{len({c['university'] for c in chunks})}** universities")

# -----------------------------------------------------------------------------
# 5. MAIN CONTENT
# -----------------------------------------------------------------------------
st.markdown("""
<div class="hero">
    <div class="hero-badge">✨ AI-Powered Admissions Guide</div>
    <h1>Find the path that fits your future.</h1>
    <p>Explore university programs, eligibility, fees and deadlines with AI answers grounded in official sources.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="warn">
    <b>⚠️ Important:</b> Raah helps you compare options. It can be wrong or outdated — always verify on the official university website. You make the final decision.
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["💬 Ask Raah", "📌 Build My Shortlist"])

# --- TAB 1: ASK RAAH ---
with tab1:
    st.markdown("### What would you like to know?")
    st.markdown("<p style='font-size: 0.95rem; margin-bottom: 1rem;'>Ask about eligibility, fees, deadlines or a university program.</p>", unsafe_allow_html=True)
    
    q = st.text_input("Ask Raah", placeholder="e.g., What are the eligibility requirements for BS Computer Science?", label_visibility="collapsed")
    
    st.markdown("<p style='font-size: 0.85rem; font-weight: 600; margin-bottom: 8px;'>Suggested questions:</p>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("📅 Admission Deadlines"):
            st.session_state.q_input = "What are the upcoming admission deadlines?"
    with c2:
        if st.button("💰 Fee Structures"):
            st.session_state.q_input = "What is the fee structure for engineering programs?"
    with c3:
        if st.button("🎓 Scholarships"):
            st.session_state.q_input = "Are there any merit-based scholarships available?"
    with c4:
        if st.button("📋 Eligibility"):
            st.session_state.q_input = "What is the eligibility criteria for Medical colleges?"

    st.markdown("<br>", unsafe_allow_html=True) 
    _, col_btn = st.columns([3, 1])
    with col_btn:
        ask_clicked = st.button("Ask Raah ➔", type="primary", use_container_width=True)

    query_to_run = st.session_state.get('q_input', q)
    
    if ask_clicked or (st.session_state.get('q_input') and not q):
        if query_to_run:
            hits = retrieve(query_to_run, f_field, f_city)
            if not hits:
                st.info(NOT_FOUND)
            else:
                ctx = "\n\n".join(f"[{h['university']} | {h['section']} | {h['year']}] {h['text']}" for h in hits)
                with st.spinner("Thinking..."):
                    try:
                        st.markdown("### Answer")
                        st.write(llm(f"Context:\n{ctx}\n\nQuestion:\n{query_to_run}"))
                        show_sources(hits)
                    except Exception as e:
                        st.error(f"The AI service failed ({e}). Please try again.")
            if 'q_input' in st.session_state:
                del st.session_state.q_input

# --- TAB 2: BUILD MY SHORTLIST ---
with tab2:
    st.markdown("### Build Your Shortlist")
    st.markdown("<p style='font-size: 0.95rem; margin-bottom: 1rem;'>Tell us about your profile and we'll suggest options from the knowledge base.</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        interest = st.text_input("Interests / Dream Career", placeholder="e.g., software, AI, business...")
        marks = st.number_input("FSc/A-level Marks (%)", 0, 100, 70)
    with col2:
        budget = st.text_input("Yearly Budget (PKR)", placeholder="e.g., 300000")
        pref_city = st.text_input("Preferred City", placeholder="e.g., Islamabad")

    st.markdown("<br>", unsafe_allow_html=True)
    suggest_clicked = st.button("✨ Suggest Options", type="primary")

    if suggest_clicked:
        if not interest:
            st.warning("Please enter your interests to get suggestions.")
        else:
            hits = retrieve(f"{interest} program eligibility fees {pref_city}", f_field, f_city, k=5)
            if not hits:
                st.info(NOT_FOUND)
            else:
                ctx = "\n\n".join(f"[{h['university']} | {h['section']} | {h['year']}] {h['text']}" for h in hits)
                profile = f"Interests: {interest}; marks: {marks}%; budget: {budget}; city: {pref_city}"
                msg = (f"Context:\n{ctx}\n\nStudent profile: {profile}\n\nSuggest up to 3 options using ONLY the context. "
                       "For each: name, why it fits, fees/deadline if stated, and what is uncertain or missing. "
                       "Say if the student may not meet the eligibility. Remind them to verify officially.")
                with st.spinner("Comparing options..."):
                    try:
                        st.markdown("### Suggested Options")
                        st.write(llm(msg))
                        show_sources(hits)
                    except Exception as e:
                        st.error(f"The AI service failed ({e}). Please try again.")
    else:
        st.markdown("""
        <div class="empty-state">
            <h3>Your shortlist is empty</h3>
            <p>Fill in your details above and click "Suggest Options" to get started.</p>
        </div>
        """, unsafe_allow_html=True)
