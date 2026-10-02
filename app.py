import os, re, glob
import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq

st.set_page_config(page_title="Raah", page_icon="🧭", layout="wide")
st.markdown("""<style>
.hero{background:linear-gradient(90deg,#0f766e,#1d4ed8);padding:22px;border-radius:14px;color:#fff;margin-bottom:14px}
.hero h1{margin:0;color:#fff}.hero p{margin:4px 0 0;opacity:.9}
.src{border-left:4px solid #0f766e;background:#f1f5f9;padding:8px 12px;border-radius:6px;margin:6px 0;color:#0f172a}
.warn{background:#fef3c7;color:#78350f;padding:8px 12px;border-radius:8px;font-size:.9rem}
</style>
<div class="hero"><h1>🧭 Raah</h1><p>AI career pathway navigator for Pakistani students - answers come only from official sources.</p></div>
<div class="warn">Raah helps you compare options. It can be wrong or outdated - always verify on the official university website. You make the final decision.</div>
""", unsafe_allow_html=True)

def secret(name, default=None):
    try:
        return st.secrets[name]
    except Exception:
        return os.getenv(name, default)

MODEL = secret("GROQ_MODEL", "llama-3.3-70b-versatile")  # check console.groq.com for current model names
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
st.sidebar.header("Filters (metadata)")
f_field = st.sidebar.selectbox("Field", fields)
f_city = st.sidebar.selectbox("City", cities)
st.sidebar.caption(f"{len(chunks)} chunks from {len({c['university'] for c in chunks})} universities")

tab1, tab2 = st.tabs(["Ask Raah", "Build my shortlist"])

with tab1:
    q = st.text_input("Ask about eligibility, fees, deadlines...", placeholder="What are the eligibility requirements for BS Computer Science?")
    if st.button("Ask", type="primary") and q:
        hits = retrieve(q, f_field, f_city)
        if not hits:
            st.info(NOT_FOUND)
        else:
            ctx = "\n\n".join(f"[{h['university']} | {h['section']} | {h['year']}] {h['text']}" for h in hits)
            with st.spinner("Thinking..."):
                try:
                    st.write(llm(f"Context:\n{ctx}\n\nQuestion:\n{q}"))
                    show_sources(hits)
                except Exception as e:
                    st.error(f"The AI service failed ({e}). Please try again.")

with tab2:
    c1, c2 = st.columns(2)
    interest = c1.text_input("Interests / dream career", placeholder="software, AI, business...")
    marks = c1.number_input("FSc/A-level marks (%)", 0, 100, 70)
    budget = c2.text_input("Yearly budget (PKR)", placeholder="e.g. 300000")
    pref_city = c2.text_input("Preferred city", placeholder="Islamabad")
    if st.button("Suggest options") and interest:
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
