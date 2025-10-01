# RAG Agriculture Chatbot (simple)

Quick local RAG prototype:
- Scrapes a URL (gets up to 5 conversation-like items),
- Builds chunks and embeddings with `all-MiniLM-L6-v2`,
- Retrieves best chunks using cosine similarity,
- Sends context+question to an LLM (replace wrapper to match your endpoint),
- Minimal Streamlit front-end.

Usage:
1. Create venv, install requirements
2. Set OPENAI_API_KEY env var or paste key in the sidebar
3. `streamlit run app.py`
