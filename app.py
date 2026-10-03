import os, re, glob
import pandas as pd
import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq

st.set_page_config(page_title="Raah - Career Navigator", page_icon="🧭", layout="wide")
S = st.session_state
for k, v in {"history": [], "shortlist": "", "cl": [], "q": "", "page": "🏠 Home"}.items():
    S.setdefault(k, v)

# ---------- Styling (palette + accessibility + light/dark) ----------
dark = bool(S.get("dark"))
P = ({"BG": "#162526", "TX": "#d5f4e6", "SB": "#0e1c1d", "INBG": "#2d4a4b", "INTX": "#ffffff", "CARD": "#1f3637", "PH": "#a9c9c3"} if dark else
     {"BG": "#d5f4e6", "TX": "#2d4a4b", "SB": "#2d4a4b", "INBG": "#ffffff", "INTX": "#2d4a4b", "CARD": "#eafaf2", "PH": "#6b8a8b"})
CSS = """<style>
html{font-size:%FS%}
.stApp{background:%BG%;color:%TX%}
.stApp p,.stApp li,.stApp label,.stApp h1,.stApp h2,.stApp h3,.stApp h4,.stApp [data-testid="stCaptionContainer"]{color:%TX%}
[data-testid="stHeader"]{background:transparent}
[data-testid="stHeader"] *{color:%TX%!important}
input,textarea{background:%INBG%!important;color:%INTX%!important;-webkit-text-fill-color:%INTX%!important}
::placeholder{color:%PH%!important;-webkit-text-fill-color:%PH%!important;opacity:1}
[data-baseweb="input"],[data-baseweb="base-input"],[data-baseweb="textarea"]{background:%INBG%!important}
.stApp div[data-baseweb="select"] *{background:%INBG%!important;color:%INTX%!important}
.stApp div[data-baseweb="select"] svg{fill:%INTX%!important}
.stTabs [data-baseweb="tab"] p{color:%TX%!important}
[data-testid="stExpander"] details{background:%CARD%;border-color:%TX%}
[data-testid="stExpander"] summary *{color:%TX%!important}
[data-testid="stChatMessage"]{background:%CARD%;border-radius:10px}
.stApp table,.stApp th,.stApp td{background:%CARD%!important;color:%TX%!important;border-color:%TX%!important}
.stApp div[data-baseweb="select"] span[data-baseweb="tag"],.stApp div[data-baseweb="select"] span[data-baseweb="tag"] *{background:#618685!important;color:#ffffff!important}
[data-baseweb="popover"] ul,[data-baseweb="popover"] li{background:#ffffff!important}
[data-baseweb="popover"] *{color:#2d4a4b!important}
[data-baseweb="popover"] li:hover,[data-baseweb="popover"] li[aria-selected="true"]{background:#d5f4e6!important}
.stButton>button,.stDownloadButton>button{background:#618685;color:#fff;border:0;border-radius:8px;font-weight:600}
.stButton>button:hover,.stDownloadButton>button:hover{background:#2d4a4b;color:#fff}
.stButton>button *,.stDownloadButton>button *{color:#fff!important}
.hero{background:linear-gradient(135deg,#2d4a4b 0%,#3f6b6c 100%);border-radius:20px;padding:36px;margin-bottom:16px;box-shadow:0 8px 24px rgba(0,0,0,.25)}
.hero .title{color:#ffffff!important;font-size:2.8rem;font-weight:800;line-height:1.2;margin:10px 0}
.hero .hero-body{background:#618685;color:#ffffff!important;padding:12px 16px;border-radius:10px}
.hero .badge{background:#fefbd8;color:#2d4a4b!important;padding:4px 12px;border-radius:999px;font-weight:700;font-size:.85rem}
.warn{background:#fefbd8;color:#2d4a4b!important;border-left:6px solid #2d4a4b;padding:12px 16px;border-radius:8px;margin-bottom:12px}
.warn *{color:#2d4a4b!important}
.src{background:#fefbd8;border-left:5px solid #618685;padding:8px 12px;border-radius:6px;margin:6px 0}
.src,.src *{color:#2d4a4b!important}
.src a{font-weight:700;text-decoration:underline}
[data-testid="stSidebar"]{background:%SB%}
[data-testid="stSidebar"] *{color:#ffffff!important}
section[data-testid="stSidebar"] div[data-baseweb="select"] *{background:#ffffff!important;color:#2d4a4b!important}
section[data-testid="stSidebar"] div[data-baseweb="select"] svg{fill:#2d4a4b!important}
.sbh{font-size:1.15rem;font-weight:700;margin:14px 0 6px}
:focus-visible{outline:3px solid #618685!important;outline-offset:2px}
[data-testid="stSidebar"] :focus-visible{outline-color:#fefbd8!important}
[class*="st-key-card_"] button{width:100%;min-height:185px;display:flex;flex-direction:column;align-items:flex-start;justify-content:flex-start;text-align:left;background:%CARD%;border:1px solid #618685;border-left:6px solid #618685;border-radius:18px;padding:20px 22px;box-shadow:0 3px 10px rgba(0,0,0,.10);transition:transform .2s,box-shadow .2s,border-color .2s}
[class*="st-key-card_"] button:hover{transform:translateY(-5px);box-shadow:0 12px 26px rgba(0,0,0,.22);border-color:%TX%;background:%CARD%}
[class*="st-key-card_"] button>div,[class*="st-key-card_"] button [data-testid="stMarkdownContainer"]{width:100%}
.stApp [class*="st-key-card_"] button *{color:%TX%!important;text-align:left!important}
.stApp [class*="st-key-card_"] button p{white-space:pre-line;margin:0;width:100%;font-size:.98rem;line-height:1.55;font-weight:400}
.stApp [class*="st-key-card_"] button p::first-line{font-size:2.4rem;line-height:1.35}
.stApp [class*="st-key-card_"] button strong{font-size:1.3rem;font-weight:800;line-height:1.7}
[data-testid="stFormSubmitButton"] button{background:#618685;color:#fff;border:0;border-radius:8px;font-weight:600}
[data-testid="stFormSubmitButton"] button *{color:#fff!important}
[data-testid="stForm"]{border:1px solid #618685;background:%CARD%;border-radius:14px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px;margin:14px 0}
.fcard{background:%CARD%;border:1px solid #618685;border-radius:16px;padding:16px;transition:transform .2s}
.fcard:hover{transform:translateY(-4px)}
.fcard .ic{font-size:1.8rem}
.fcard h4{margin:.3rem 0;color:%TX%!important}
.fcard p{margin:0;color:%TX%!important}
.st-key-fab{position:fixed;right:22px;bottom:22px;z-index:1001;width:auto!important}
.st-key-fab button{width:62px;height:62px;border-radius:50%;font-size:1.6rem;box-shadow:0 6px 20px rgba(0,0,0,.35)}
.st-key-chatbox{position:fixed;right:22px;bottom:96px;width:min(420px,92vw);max-height:68vh;overflow-y:auto;background:%BG%;border:2px solid #618685;border-radius:16px;padding:14px;z-index:1000;box-shadow:0 10px 30px rgba(0,0,0,.4)}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
</style>"""
css = CSS.replace("%FS%", "125%" if S.get("large") else "100%")
for k, v in P.items():
    css = css.replace("%" + k + "%", v)
st.markdown(css, unsafe_allow_html=True)
st.markdown("""<div class="hero" role="banner"><span class="badge">Grounded in official sources</span>
<div class="title" role="heading" aria-level="1">Find the path that fits your future.</div>
<div class="hero-body">Explore university programs, eligibility, fees and deadlines with AI answers grounded in official sources.</div></div>
<div class="warn" role="alert"><b>Important:</b> Raah helps you compare options. It can be wrong or outdated - always verify on the official university website. You make the final decision.</div>""", unsafe_allow_html=True)

# ---------- Config ----------
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

# ---------- Data + index ----------
def load_chunks():
    chunks = []
    for path in sorted(glob.glob("data/*.txt")):
        if "sample_template" in path:
            continue
        head, _, body = open(path, encoding="utf-8").read().partition("\n\n")
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
                           "content": content, "university": meta.get("university", "?"), "city": meta.get("city", "?"),
                           "field": meta.get("field", "?"), "year": meta.get("year", "?"),
                           "source": meta.get("source", ""), "section": section})
    return chunks

@st.cache_resource(show_spinner="Building knowledge base (one time)...")
def build_index(version="v3"):
    chunks = load_chunks()
    if not chunks:
        return None, None, []
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    emb = model.encode([c["text"] for c in chunks], normalize_embeddings=True).astype("float32")
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    return model, index, chunks

model, index, chunks = build_index()
if not chunks:
    st.error("No documents found. Add .txt files to the data/ folder.")
    st.stop()

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

def llm(msg, system=SYSTEM):
    r = Groq(api_key=secret("GROQ_API_KEY")).chat.completions.create(
        model=MODEL, temperature=0.1, messages=[{"role": "system", "content": system}, {"role": "user", "content": msg}])
    return r.choices[0].message.content.strip()

def ctx_of(hits):
    return "\n\n".join(f"[{h['university']} | {h['section']} | {h['year']}] {h['text']}" for h in hits)

def sources(hits):
    with st.expander("Sources used"):
        for h in hits:
            st.markdown(f'<div class="src"><b>{h["university"]}</b> - {h["section"]} ({h["field"]}, {h["city"]}, {h["year"]})<br>'
                        f'<a href="{h["source"]}" target="_blank">Open official source</a> - match score {h["score"]:.2f}</div>',
                        unsafe_allow_html=True)

def run_query(q, multi, field, city):
    trace, sq = [], q
    if multi:
        try:
            sq = llm("Rewrite this student question as a short search query (max 15 words). Output only the query.\n" + q,
                     "You are a query planner.")
            trace.append(("Planner", f"Search query: {sq}"))
        except Exception:
            sq = q
    hits = retrieve(sq, field, city) or (retrieve(q, field, city) if sq != q else [])
    trace.append(("Retriever", f"{len(hits)} relevant chunks found"))
    if not hits:
        return NOT_FOUND, [], trace
    ans = llm(f"Context:\n{ctx_of(hits)}\n\nQuestion:\n{q}")
    trace.append(("Answerer", "Draft answer written from sources"))
    if multi:
        try:
            ans = llm(f"Context:\n{ctx_of(hits)}\n\nDraft answer:\n{ans}\n\nCheck every claim against the context. Rewrite the answer "
                      "keeping ONLY supported claims. If something is unsupported, remove it and say it was not found. Output only the final answer.")
            trace.append(("Reviewer", "Claims checked against sources"))
        except Exception:
            trace.append(("Reviewer", "Skipped (service error)"))
    return ans, hits, trace

# ---------- Sidebar ----------
st.sidebar.markdown('<div class="sbh" style="font-size:1.5rem">🧭 Raah</div><div>Career Navigator</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sbh">Refine your search</div>', unsafe_allow_html=True)
f_field = st.sidebar.selectbox("Field of study", ["All"] + sorted({c["field"] for c in chunks}))
f_city = st.sidebar.selectbox("City", ["All"] + sorted({c["city"] for c in chunks}))
multi = st.sidebar.toggle("Multi-agent mode (Planner + Reviewer)", key="multi", help="Slower but double-checks answers against sources.")
st.sidebar.markdown('<div class="sbh">Display &amp; accessibility</div>', unsafe_allow_html=True)
st.sidebar.toggle("🌙 Dark mode", key="dark")
st.sidebar.checkbox("Large text", key="large")
ups = sum(1 for h in S.history if h.get("fb") == "up"); downs = sum(1 for h in S.history if h.get("fb") == "down")
st.sidebar.caption(f"📊 {len(chunks)} chunks from {len({c['university'] for c in chunks})} universities")
st.sidebar.caption(f"Feedback this session: 👍 {ups}  👎 {downs}")

P_HOME, P_ASK, P_CMP, P_SL, P_CL = "🏠 Home", "💬 Ask Raah", "⚖️ Compare", "📌 Shortlist", "✅ Checklist"
PAGES = [P_HOME, P_ASK, P_CMP, P_SL, P_CL]
st.radio("Navigate", PAGES, key="page", horizontal=True, label_visibility="collapsed")

def go(p): S.page = p
def go_multi(): S.multi = True; S.page = P_ASK
def flip_dark(): S.dark = not S.get("dark", False)

if S.page == P_HOME:
    cards = [("card_ask", "💬\n**Ask Raah**\nSource-cited answers, with a safe \"not found\" instead of guessing.", go, (P_ASK,)),
             ("card_cmp", "⚖️\n**Compare**\nEligibility, fees and deadlines side by side.", go, (P_CMP,)),
             ("card_sl", "📌\n**Shortlist**\nOptions matched to your marks, interests and budget.", go, (P_SL,)),
             ("card_cl", "✅\n**Checklist**\nApplication steps you approve, edit or reject.", go, (P_CL,)),
             ("card_multi", "🤖\n**Multi-agent**\nPlanner and Reviewer agents double-check answers. Tap to turn it on and ask.", go_multi, ()),
             ("card_dark", "♿\n**Accessible**\nHigh contrast, large text, keyboard friendly. Tap to switch dark mode.", flip_dark, ())]
    for row in (cards[:3], cards[3:]):
        for col, (k, label, fn, args) in zip(st.columns(3), row):
            col.button(label, key=k, on_click=fn, args=args, use_container_width=True)
    st.caption("Tap any card to open it.")

# ---------- Tab 2: Compare ----------
if S.page == P_CMP:
    st.subheader("Compare universities side by side")
    unis = sorted({c["university"] for c in chunks})
    pick = st.multiselect("Choose 2-3 universities", unis, max_selections=3)
    def sec(u, f, key):
        m = [c for c in chunks if c["university"] == u and c["field"] == f and key in c["section"].lower()]
        return " ".join(c.get("content") or c["text"].split(": ", 1)[-1] for c in m) or "Not in knowledge base"
    rows = []
    for u in pick:
        for f in sorted({c["field"] for c in chunks if c["university"] == u}):
            if f_field != "All" and f != f_field:
                continue
            src = next(c["source"] for c in chunks if c["university"] == u and c["field"] == f)
            rows.append({"University": u, "Field": f, "Eligibility": sec(u, f, "eligib"), "Fees": sec(u, f, "fee"),
                         "Deadlines": sec(u, f, "deadline"), "Source": src})
    if rows:
        st.table(pd.DataFrame(rows).set_index("University"))
        st.caption("Taken directly from your indexed documents - verify on the official sites.")
    else:
        st.info("Select universities to compare.")

# ---------- Tab 3: Shortlist ----------
if S.page == P_SL:
    st.subheader("Build my shortlist")
    c1, c2 = st.columns(2)
    interest = c1.text_input("Interests / dream career", placeholder="software, AI, business...")
    marks = c1.number_input("FSc / A-level marks (%)", 0, 100, 70)
    budget = c2.text_input("Yearly budget (PKR)", placeholder="e.g. 300000")
    pref_city = c2.text_input("Preferred city", placeholder="Islamabad")
    if st.button("Suggest options") and interest:
        hits = retrieve(f"{interest} program eligibility fees {pref_city}", f_field, f_city, k=5)
        if not hits:
            S.shortlist = NOT_FOUND
        else:
            with st.spinner("Comparing options..."):
                try:
                    S.shortlist = llm(f"Context:\n{ctx_of(hits)}\n\nStudent profile: interests {interest}; marks {marks}%; budget {budget}; city {pref_city}\n\n"
                                      "Suggest up to 3 options using ONLY the context. For each: name, why it fits, fees/deadline if stated, what is uncertain. "
                                      "Say if the student may not meet eligibility. Remind them to verify officially.")
                    S.shortlist_hits = hits
                except Exception as e:
                    st.error(f"The AI service failed ({e}).")
    if S.shortlist:
        st.write(S.shortlist)
        if S.get("shortlist_hits") and S.shortlist != NOT_FOUND:
            sources(S.shortlist_hits)
        st.download_button("⬇ Download shortlist", S.shortlist + "\n\nAlways verify on official university websites.", file_name="raah_shortlist.txt")

# ---------- Tab 4: Checklist ----------
if S.page == P_CL:
    st.subheader("Application checklist (you approve, edit or reject each step)")
    u = st.selectbox("University", sorted({c["university"] for c in chunks}), key="cl_u")
    f = st.selectbox("Field", sorted({c["field"] for c in chunks if c["university"] == u}), key="cl_f")
    if st.button("Generate checklist"):
        part = [c for c in chunks if c["university"] == u and c["field"] == f]
        with st.spinner("Preparing..."):
            try:
                out = llm(f"Context:\n{ctx_of([{**c, 'score': 0} for c in part])}\n\nCreate a checklist of 5-10 application steps using ONLY the context. "
                          "One step per line starting with '- '. Include deadlines, documents and fees only if stated. Do not invent anything.")
                S.cl = [l.lstrip("-• ").strip() for l in out.splitlines() if l.strip().startswith(("-", "•"))]
                S.cl_src = part[0]["source"]
                for k in [k for k in S if str(k).startswith(("cl_t", "cl_s"))]:
                    del S[k]
            except Exception as e:
                st.error(f"The AI service failed ({e}).")
    approved = []
    for i, item in enumerate(S.cl):
        a, b = st.columns([4, 2])
        txt = a.text_input(f"Step {i+1}", value=item, key=f"cl_t{i}")
        stat = b.radio(f"Decision for step {i+1}", ["Approve", "Reject"], horizontal=True, key=f"cl_s{i}")
        if stat == "Approve":
            approved.append(txt)
    if S.cl:
        st.caption(f"{len(approved)} of {len(S.cl)} steps approved. Verify at: {S.get('cl_src', '')}")
        st.download_button("⬇ Download approved checklist", "\n".join(f"[ ] {t}" for t in approved), file_name="raah_checklist.txt")

# ---------- Ask Raah (main area) ----------
def vote(i, kind): S.history[i]["fb"] = kind
def queue(t): S["pending"] = t

if S.page == P_ASK:
    st.subheader("What would you like to know?")
    with st.form("askform", clear_on_submit=True):
        qv = st.text_input("Your question (press Enter to send)", placeholder="e.g., What are the eligibility requirements for BS Computer Science?")
        sent = st.form_submit_button("Ask Raah →")
    st.write("Suggested questions:")
    quick = {"Admission deadlines": "What are the admission deadlines?", "Fee structures": "What are the fee structures?",
             "Scholarships": "What scholarships are available?", "Eligibility": "What are the eligibility requirements?"}
    for col, (label, text) in zip(st.columns(4), quick.items()):
        col.button(label, on_click=queue, args=(text,), key="quick_" + label)
    query = (qv.strip() if sent else "") or S.pop("pending", "")
    if query:
        with st.spinner("Thinking..."):
            try:
                a, h, t = run_query(query, multi, f_field, f_city)
                S.history.append({"q": query, "a": a, "hits": h, "trace": t, "multi": multi, "fb": None})
            except Exception as e:
                st.error(f"The AI service failed ({e}). Please try again.")
    for i in range(len(S.history) - 1, -1, -1):
        it = S.history[i]
        with st.chat_message("user"):
            st.write(it["q"])
        with st.chat_message("assistant"):
            st.write(it["a"])
            if it["hits"]:
                sources(it["hits"])
            with st.expander("Agent steps" + (" (multi-agent)" if it["multi"] else " (single-agent)")):
                for name, info in it["trace"]:
                    st.write(f"**{name}:** {info}")
            if it["fb"]:
                st.caption("Thanks for your feedback.")
            else:
                c1, c2, _ = st.columns([1, 1, 6])
                c1.button("👍 Helpful", key=f"up{i}", on_click=vote, args=(i, "up"))
                c2.button("👎 Not helpful", key=f"dn{i}", on_click=vote, args=(i, "down"))
