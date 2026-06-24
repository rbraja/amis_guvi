
import os
import chromadb

from src.rag_utils import get_embedding
from src.config import BASE_DIR
from src.rag_loader import load_rag_data


client = chromadb.PersistentClient(path=os.path.join(BASE_DIR, "vector_db"))

collection = client.get_or_create_collection(name="amis_rag")


def _tfidf_search(question, top_k=3):
    """Fallback TF-IDF search over the RAG CSV sources.

    Returns same format as the Chroma search.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except Exception:
        return []

    documents = load_rag_data()

    texts = [d.get("text", "") for d in documents]

    if not texts:
        return []

    try:
        vectorizer = TfidfVectorizer()
        doc_vectors = vectorizer.fit_transform(texts)

        q_vec = vectorizer.transform([question])
        sims = cosine_similarity(q_vec, doc_vectors)[0]

        top_idxs = sims.argsort()[-top_k:][::-1]

        results = []
        for idx in top_idxs:
            results.append({
                "text": documents[idx]["text"],
                "source": documents[idx].get("source"),
                "score": round(float(sims[idx]) * 100, 2),
                "backend": "tfidf"
            })

        return results
    except Exception:
        return []


def search_documents(question, top_k=3):
    """Try semantic ChromaDB retrieval; if empty or failing, fall back to TF-IDF."""

    query_emb = get_embedding(question)

    try:
        results = collection.query(
            query_embeddings=[query_emb],
            n_results=top_k,
            include=["documents", "metadatas", "distances", "ids"],
        )
    except Exception:
        return _tfidf_search(question, top_k=top_k)
    documents = results.get("documents", [[]])[0]
    if not documents:
        return _tfidf_search(question, top_k=top_k)

    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    out = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        try:
            score = round(max(0.0, 1.0 - dist) * 100, 2)
        except Exception:
            score = None

        source = None
        if isinstance(meta, dict):
            source = meta.get("source")
        else:
            source = meta

        out.append({"text": doc, "source": source, "score": score, "backend": "chroma"})

    return out