import os, re, glob
import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq

st.set_page_config(page_title="Raah", page_icon="🧭", layout="wide")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@600;700;800&display=swap');
html, body, [class*="css"] {font-family:'DM Sans',sans-serif;}
.stApp{background:radial-gradient(circle at 90% 5%,rgba(83,105,230,.10),transparent 25%),radial-gradient(circle at 5% 20%,rgba(67,190,155,.08),transparent 22%),#f7f8fc;color:#172033;}
.block-container{max-width:1180px;padding-top:2rem;padding-bottom:4rem;}
[data-testid="stSidebar"]{background:#fff;border-right:1px solid #e7eaf2;}
.brand{display:flex;align-items:center;gap:12px;margin-bottom:2rem;}
.brand-icon{width:44px;height:44px;border-radius:14px;display:flex;align-items:center;justify-content:center;background:linear-gradient(135deg,#5b73f0,#304dc9);color:#fff;font-size:22px;box-shadow:0 9px 22px rgba(53,89,224,.22);}
.brand-name{font-family:'Plus Jakarta Sans',sans-serif;font-weight:800;font-size:1.35rem;letter-spacing:-.04em}.brand-sub{color:#8a93a5;font-size:.72rem}
.hero{position:relative;overflow:hidden;padding:46px 48px;border-radius:28px;color:#fff;background:radial-gradient(circle at 85% 20%,rgba(255,255,255,.18),transparent 20%),linear-gradient(135deg,#1c2c70,#3559e0 58%,#657af5);box-shadow:0 18px 50px rgba(38,61,150,.20);margin-bottom:22px;}
.hero h1{font-family:'Plus Jakarta Sans',sans-serif;font-size:clamp(2.2rem,4vw,3.35rem);line-height:1.05;letter-spacing:-.055em;margin:0 0 12px;color:#fff}.hero p{max-width:650px;font-size:1rem;line-height:1.65;color:rgba(255,255,255,.84);margin:0}.hero-kicker{display:inline-block;padding:7px 12px;border:1px solid rgba(255,255,255,.25);background:rgba(255,255,255,.1);border-radius:999px;font-size:.76rem;font-weight:600;margin-bottom:18px;}
.notice{padding:13px 17px;border:1px solid #e4e8f1;background:#fff;border-radius:15px;color:#697386;font-size:.83rem;margin-bottom:26px;}
.section-title{font-family:'Plus Jakarta Sans',sans-serif;font-size:1.18rem;font-weight:800;letter-spacing:-.025em;margin:8px 0 5px}.section-sub{color:#687386;font-size:.88rem;margin-bottom:16px}
.stTabs [data-baseweb="tab-list"]{gap:8px;border-bottom:1px solid #e7eaf2;margin-bottom:22px}.stTabs [data-baseweb="tab"]{border-radius:12px 12px 0 0;padding:11px 18px;font-weight:600;color:#7b8495}.stTabs [aria-selected="true"]{color:#3559e0!important}
[data-testid="stTextInput"] input,[data-testid="stNumberInput"] input{border-radius:13px;border:1px solid #dfe3ec;background:#fff;padding:12px 14px;}
.stButton>button{border-radius:13px;border:0;min-height:44px;font-weight:700;transition:.2s ease}.stButton>button[kind="primary"]{background:linear-gradient(135deg,#4665e8,#314fd0);box-shadow:0 8px 20px rgba(53,89,224,.18)}.stButton>button:hover{transform:translateY(-1px)}
.answer-card{background:#fff;border:1px solid #e7eaf2;border-radius:22px;padding:26px 28px;margin-top:18px;box-shadow:0 10px 35px rgba(28,39,74,.06);line-height:1.7}.answer-label{color:#3559e0;text-transform:uppercase;letter-spacing:.12em;font-size:.68rem;font-weight:800;margin-bottom:12px}
.src{border:1px solid #e4e8f1!important;border-left:4px solid #5369e6!important;background:#fbfcff!important;padding:12px 14px!important;border-radius:12px!important;margin:9px 0!important;color:#172033!important}
.stat{background:#fff;border:1px solid #e7eaf2;border-radius:17px;padding:15px 8px;text-align:center}.stat-number{font-family:'Plus Jakarta Sans';font-size:1.4rem;font-weight:800}.stat-label{color:#8992a2;font-size:.72rem}.sidebar-title{font-family:'Plus Jakarta Sans';font-weight:800;font-size:.9rem;margin-bottom:10px}.sidebar-note{color:#8790a0;font-size:.74rem;line-height:1.5;padding-top:15px}.footer{text-align:center;color:#8a93a5;font-size:.74rem;margin-top:42px;padding-top:20px;border-top:1px solid #e7eaf2}
@media(max-width:700px){.hero{padding:30px 24px;border-radius:22px}.hero h1{font-size:2.2rem}.block-container{padding-top:1rem}}
</style>
<div class="hero"><div class="hero-kicker">✦ Your next step starts here</div><h1>Find the path that<br>fits your future.</h1><p>Explore university programs, eligibility, fees and deadlines with AI answers grounded in the Raah knowledge base.</p></div><div class="notice"><b>Built for students.</b> Raah helps you explore your options — you make the final decision. Always verify current details with the official university source.</div>""", unsafe_allow_html=True)

def secret(name, default=None):
    try:
        return st.secrets[name]
    except Exception:
        return os.getenv(name, default)

MODEL = secret("GROQ_MODEL", "openai/gpt-oss-120b")
MIN_SCORE = 0.25  # below this, we say "not found" (tune using your test questions)
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
    with st.expander("Sources used"):
        for h in hits:
            st.markdown(f'<div class="src"><b>{h["university"]}</b> - {h["section"]} ({h["field"]}, {h["city"]}, {h["year"]})<br>'
                        f'<a href="{h["source"]}">{h["source"]}</a> - match score {h["score"]:.2f}</div>', unsafe_allow_html=True)

if not chunks:
    st.error("No documents found. Add .txt files to the data/ folder (see data/sample_template.txt).")
    st.stop()

fields = ["All"] + sorted({c["field"] for c in chunks})
cities = ["All"] + sorted({c["city"] for c in chunks})
with st.sidebar:
    st.markdown("""<div class="brand"><div class="brand-icon">🧭</div><div><div class="brand-name">Raah</div><div class="brand-sub">Career Pathway Navigator</div></div></div>""", unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">Refine your search</div>', unsafe_allow_html=True)
    f_field = st.selectbox("Study field", fields)
    f_city = st.selectbox("City", cities)
    st.divider()
    uni_count=len({c["university"] for c in chunks}); field_count=len({c["field"] for c in chunks})
    a,b=st.columns(2)
    a.markdown(f'<div class="stat"><div class="stat-number">{uni_count}</div><div class="stat-label">Universities</div></div>',unsafe_allow_html=True)
    b.markdown(f'<div class="stat"><div class="stat-number">{field_count}</div><div class="stat-label">Fields</div></div>',unsafe_allow_html=True)
    st.markdown('<div class="sidebar-note">Grounded in the available university knowledge base. Verify important admission details on official websites.</div>',unsafe_allow_html=True)

tab1, tab2 = st.tabs(["✦  Ask Raah", "◈  Build My Shortlist"])

with tab1:
    st.markdown('<div class="section-title">What would you like to know?</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Ask about eligibility, fees, deadlines or a university program.</div>', unsafe_allow_html=True)
    q = st.text_input("Question", placeholder="e.g. What are the eligibility requirements for BS Computer Science at NUST?", label_visibility="collapsed")
    if st.button("Ask Raah  →", type="primary", use_container_width=True) and q:
        hits = retrieve(q, f_field, f_city)
        if not hits:
            st.info(NOT_FOUND)
        else:
            ctx = "\n\n".join(f"[{h['university']} | {h['section']} | {h['year']}] {h['text']}" for h in hits)
            with st.spinner("Thinking..."):
                try:
                    answer = llm(f"Context:\n{ctx}\n\nQuestion:\n{q}")
                    st.markdown(f"""<div class="answer-card"><div class="answer-label">✦ Raah's answer</div><div>{answer.replace(chr(10), '<br>')}</div></div>""", unsafe_allow_html=True)
                    show_sources(hits)
                except Exception as e:
                    st.error(f"The AI service failed ({e}). Please try again.")

with tab2:
    c1, c2 = st.columns(2)
    interest = c1.text_input("Interests / dream career", placeholder="software, AI, business...")
    marks = c1.number_input("FSc/A-level marks (%)", 0, 100, 70)
    budget = c2.text_input("Yearly budget (PKR)", placeholder="e.g. 300000")
    pref_city = c2.text_input("Preferred city", placeholder="Islamabad")
    if st.button("Build My Shortlist  →", type="primary", use_container_width=True) and interest:
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
                    answer = llm(msg)
                    st.markdown(f"""<div class="answer-card"><div class="answer-label">◈ Your shortlist</div><div>{answer.replace(chr(10), '<br>')}</div></div>""", unsafe_allow_html=True)
                    show_sources(hits)
                except Exception as e:
                    st.error(f"The AI service failed ({e}). Please try again.")


st.markdown('<div class="footer">Raah · AI Career Pathway Navigator · Grounded in the available university knowledge base</div>', unsafe_allow_html=True)
