# Raah - AI Career Pathway Navigator
Grounded (RAG) answers and shortlists for Pakistani students, built from official university pages.
Pipeline: documents -> section chunks + metadata -> embeddings -> FAISS -> retrieval -> grounded Groq answer + sources.
Run: `pip install -r requirements.txt` then `streamlit run app.py`. Add GROQ_API_KEY in Streamlit secrets.
Add data: one .txt per university/field in `data/` (header lines, blank line, then `## Section` blocks).
Raah informs; the student decides. Always verify on the official site.
