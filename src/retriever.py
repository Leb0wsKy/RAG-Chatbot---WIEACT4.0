# src/retriever.py
from typing import List, Tuple
import numpy as np
from .embedder import embed_texts
from .utils import chunk_text, cosine_similarity
from tqdm import tqdm

def build_corpus_from_documents(documents: List[str], chunk_words: int = 120, overlap: int = 20):
    """
    documents: list of long texts (e.g., each conversation scraped)
    returns:
      - chunks: list of chunk strings
      - metadata: list of dicts {doc_id, chunk_index}
      - embeddings: numpy array (N, D)
    """
    all_chunks = []
    metadata = []
    for doc_id, doc in enumerate(documents):
        chunks = chunk_text(doc, max_words=chunk_words, overlap_words=overlap)
        for idx, ch in enumerate(chunks):
            all_chunks.append(ch)
            metadata.append({"doc_id": doc_id, "chunk_index": idx, "length_words": len(ch.split())})
    if not all_chunks:
        return [], [], None
    embeddings = embed_texts(all_chunks)
    return all_chunks, metadata, embeddings

def retrieve_top_k(query: str, chunks: List[str], embeddings: np.ndarray, k: int = 1):
    """
    Returns list of (chunk_text, score, index) sorted by descending score.
    """
    if embeddings is None or len(chunks) == 0:
        return []
    query_emb = embed_texts([query])[0]
    scores = []
    for i, emb in enumerate(embeddings):
        score = cosine_similarity(query_emb, emb)
        scores.append((score, i))
    scores.sort(reverse=True, key=lambda x: x[0])
    results = []
    for score, idx in scores[:k]:
        results.append((chunks[idx], float(score), idx))
    return results
