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

    /* Global Font & Background */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background-color: #F8FAFC;
    }

    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* --- HERO SECTION --- */
    .hero {
        background: linear-gradient(135deg, #0f766e 0%, #1d4ed8 100%);
        padding: 3rem 2rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    .hero-badge {
        background-color: rgba(255, 255, 255, 0.2);
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 1rem;
        letter-spacing: 0.5px;
    }
    .hero h1 {
        font-size: 2.8rem;
        font-weight: 700;
        margin: 0 0 1rem 0;
        color: white;
    }
    .hero p {
        font-size: 1.1rem;
        max-width: 600px;
        margin: 0 auto;
        opacity: 0.9;
        line-height: 1.6;
    }

    /* --- WARNING BANNER --- */
    .warn {
        background: #FFFBEB;
        border-left: 4px solid #F59E0B;
        color: #78350F;
        padding: 1rem;
        border-radius: 8px;
        font-size: 0.85rem;
        margin-bottom: 2rem;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    }

    /* --- SIDEBAR --- */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E2E8F0;
        padding-top: 2rem;
    }
    .logo-area {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 2rem;
    }
    .logo-icon {
        background-color: #0f766e;
        color: white;
        width: 40px;
        height: 40px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
    }
    .logo-text h2 { margin: 0; font-size: 1.2rem; color: #1E293B; }
    .logo-text p { margin: 0; font-size: 0.8rem; color: #64748B; }

    /* --- INPUTS & BUTTONS --- */
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 1px solid #CBD5E1;
        padding: 12px 16px;
        font-size: 1rem;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    }
    .stTextInput > div > div > input:focus {
        border-color: #3B82F6;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2);
    }
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        padding: 10px 24px;
        transition: all 0.2s;
    }
    .stButton > button[kind="primary"] {
        background-color: #1d4ed8;
        border: none;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #1e40af;
        box-shadow: 0 4px 6px -1px rgba(29, 78, 216, 0.3);
    }

    /* --- SOURCE CARDS --- */
    .src {
        border-left: 4px solid #0f766e;
        background: #F8FAFC;
        padding: 12px 16px;
        border-radius: 6px;
        margin: 8px 0;
        color: #0f172a;
        font-size: 0.85rem;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    }
    .src a { color: #1d4ed8; text-decoration: none; font-weight: 500;}
    .src a:hover { text-decoration: underline; }

    /* --- SHORTLIST CARDS --- */
    .uni-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .uni-card h3 { margin-top: 0; color: #1E293B; font-size: 1.2rem; }
    .uni-tag {
        display: inline-block;
        background: #DBEAFE;
        color: #1E40AF;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 8px;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. BACKEND LOGIC (Unchanged)
# -----------------------------------------------------------------------------
def secret(name, default=None):
    try:
        return st.secrets[name]
    except Exception:
        return os.getenv(name, default)

MODEL = secret("GROQ_MODEL", "llama-3.3-70b-versatile")
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
            <p>Career Pathway Navigator</p>
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
# Hero Section
st.markdown("""
<div class="hero">
    <div class="hero-badge">✨ AI-Powered Admissions Guide</div>
    <h1>Find the path that fits your future.</h1>
    <p>Explore university programs, eligibility, fees and deadlines with AI answers grounded in official sources.</p>
</div>
""", unsafe_allow_html=True)

# Warning Banner
st.markdown("""
<div class="warn">
    <b>⚠️ Important:</b> Raah helps you compare options. It can be wrong or outdated — always verify on the official university website. You make the final decision.
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2 = st.tabs(["💬 Ask Raah", "📌 Build My Shortlist"])

# --- TAB 1: ASK RAAH ---
with tab1:
    st.markdown("### What would you like to know?")
    st.markdown("<p style='color: #64748B; font-size: 0.9rem;'>Ask about eligibility, fees, deadlines or a university program.</p>", unsafe_allow_html=True)
    
    q = st.text_input("Ask Raah", placeholder="e.g., What are the eligibility requirements for BS Computer Science?", label_visibility="collapsed")
    
    # Quick Prompt Chips
    st.markdown("<p style='font-size: 0.8rem; color: #64748B; margin-bottom: 5px;'>Suggested questions:</p>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("📅 Admission Deadlines", use_container_width=True):
            st.session_state.q_input = "What are the upcoming admission deadlines?"
    with c2:
        if st.button("💰 Fee Structures", use_container_width=True):
            st.session_state.q_input = "What is the fee structure for engineering programs?"
    with c3:
        if st.button("🎓 Scholarships", use_container_width=True):
            st.session_state.q_input = "Are there any merit-based scholarships available?"
    with c4:
        if st.button("📋 Eligibility", use_container_width=True):
            st.session_state.q_input = "What is the eligibility criteria for Medical colleges?"

    # Submit Button aligned right
    _, col_btn = st.columns([4, 1])
    with col_btn:
        ask_clicked = st.button("Ask Raah ➔", type="primary", use_container_width=True)

    # Handle Ask Logic
    # Check if a chip was clicked or the button was pressed
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
                        st.write(llm(f"Context:\n{ctx}\n\nQuestion:\n{query_to_run}"))
                        show_sources(hits)
                    except Exception as e:
                        st.error(f"The AI service failed ({e}). Please try again.")
            # Clear the chip state after running
            if 'q_input' in st.session_state:
                del st.session_state.q_input

# --- TAB 2: BUILD MY SHORTLIST ---
with tab2:
    st.markdown("### Build Your Shortlist")
    st.markdown("<p style='color: #64748B; font-size: 0.9rem;'>Tell us about your profile and we'll suggest options from the knowledge base.</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        interest = st.text_input("Interests / Dream Career", placeholder="e.g., software, AI, business...")
        marks = st.number_input("FSc/A-level Marks (%)", 0, 100, 70)
    with col2:
        budget = st.text_input("Yearly Budget (PKR)", placeholder="e.g., 300000")
        pref_city = st.text_input("Preferred City", placeholder="e.g., Islamabad")

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
                        st.write(llm(msg))
                        show_sources(hits)
                    except Exception as e:
                        st.error(f"The AI service failed ({e}). Please try again.")
    else:
        # Empty State Placeholder
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background: white; border-radius: 12px; border: 1px dashed #CBD5E1; margin-top: 2rem;">
            <h3 style="color: #64748B; margin-bottom: 0.5rem;">Your shortlist is empty</h3>
            <p style="color: #94A3B8; font-size: 0.9rem;">Fill in your details above and click "Suggest Options" to get started.</p>
        </div>
        """, unsafe_allow_html=True)
