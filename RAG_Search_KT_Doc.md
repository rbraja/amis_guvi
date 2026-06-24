# RAG Search KT & Demo Document

## 1. Overview
This project implements a small RAG assistant for apartment maintenance support. The assistant uses:
- CSV knowledge sources (`society_rules.csv`, `service_logs.csv`)
- PDF manuals in `data/asset_manuals` and `data/rag/asset_manuals`
- A semantic vector store (ChromaDB)
- A TF-IDF fallback search when embeddings are unavailable or fail

The RAG assistant is exposed through the Streamlit page `pages/12_RAG_Assistant.py`.

## 2. Libraries Used

Core runtime libraries:
- `streamlit` — UI framework for the front-end app
- `pandas` — CSV loading and data handling
- `numpy` — numerical operations
- `scikit-learn` — TF-IDF vectorization and cosine similarity fallback
- `chromadb` — vector database for semantic retrieval
- `sentence-transformers` — embedding model loader
- `pypdf` — PDF text extraction
- `transformers`, `torch`, `datasets`, `sentencepiece`, `accelerate` — required by text / transformer model stack

## 3. Model Used
The project uses a sentence-transformers embedding model:
- `all-MiniLM-L6-v2`

This model is loaded in `src/rag_utils.py`:
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

def get_embedding(text):
    return model.encode(
        str(text)
    ).tolist()
```

The model converts text into a numeric vector used by ChromaDB for semantic search.

## 4. Key Files and Their Roles

- `src/config.py`
  - Defines project path constants: `BASE_DIR`, `DATA_DIR`, `MODEL_DIR`, `REPORT_DIR`

- `src/rag_utils.py`
  - Loads the embedding model and exposes `get_embedding(text)`

- `src/rag_setup.py`
  - Builds / updates the vector database
  - Loads CSV sources and PDF sources
  - Adds documents and embeddings into ChromaDB

- `src/rag_loader.py`
  - Loads the same knowledge sources for TF-IDF fallback search
  - Includes PDF text extraction for fallback search

- `src/rag_search.py`
  - Runs the user query search pipeline
  - Uses Chroma semantic retrieval first
  - Falls back to TF-IDF search if Chroma fails or returns no documents

- `pages/12_RAG_Assistant.py`
  - Streamlit UI page that triggers the search
  - Includes a `Rebuild Vector DB` button and a text input for questions

## 5. Data Sources and Search Coverage

The search pipeline uses these sources:
1. `data/rag/society_rules.csv`
2. `data/rag/service_logs.csv` (column: `repair_notes`)
3. PDF files in:
   - `data/rag`
   - `data/asset_manuals`
   - `data/rag/asset_manuals`

Note: The code was updated to ensure `data/rag/asset_manuals` is also scanned, which fixes missing PDF ingestion from that nested location.

## 6. RAG Build Flow (Vector DB Creation)

Run:
```bash
python -m src.rag_setup
```

Inside `src/rag_setup.py`:
1. Create or open ChromaDB persistent client in `vector_db`
2. Call `add_society_rules(collection, rules_path)`
   - Reads `society_rules.csv`
   - Converts each row to text
   - Embeds text and writes to ChromaDB
3. Call `add_service_logs(collection, logs_path)`
   - Reads `service_logs.csv`
   - Uses `repair_notes` text
   - Embeds and stores records in ChromaDB
4. Call `add_pdf_documents(collection, pdf_dirs)`
   - Recursively scans configured PDF directories
   - Extracts PDF text using `pypdf`
   - Embeds and stores each PDF document

The result is a collection named `amis_rag` containing semantic documents with metadata sources.

## 7. Search Flow Step-by-Step

### 7.1 User query triggers search
The Streamlit page calls `search_documents(question, top_k)` in `src/rag_search.py`.

### 7.2 Semantic retrieval
In `src/rag_search.py`:
1. `query_emb = get_embedding(question)`
2. `collection.query(...)` against ChromaDB
3. Extract returned documents, metadata, and distances
4. Convert distance into a score and return results labeled `backend: "chroma"`

If Chroma returns no results or the query raises an exception, the code falls back to TF-IDF.

### 7.3 TF-IDF fallback
Fallback search uses `_tfidf_search(question, top_k)`:
1. Load documents from `src.rag_loader.py` via `load_rag_data()`
   - Reads CSV rows and turns each row into text
   - Reads PDF files from the same scan paths
2. Vectorize documents via `TfidfVectorizer`
3. Compute cosine similarity between the query and all doc vectors
4. Return top matches, labeled `backend: "tfidf"`

### 7.4 What is returned
Each result includes:
- `text` — source document content
- `source` — relative file or CSV name
- `score` — relevance score
- `backend` — `chroma` or `tfidf`

## 8. Streamlit RAG Assistant Demo Flow

In `pages/12_RAG_Assistant.py`:
- User enters a question in a text input
- User selects `top_k` results
- The app calls `search_documents(question, top_k)`
- Results are displayed with source and score
- The page includes a button to rebuild the vector DB via `src.rag_setup.main()`

This is the complete user-facing search/demo experience.

## 9. Troubleshooting Notes

### PDF not fetched in RAG search
If PDFs are missing in search results, verify:
- `python -m src.rag_setup` was run after adding PDFs
- the PDF directories include `data/rag/asset_manuals`
- the vector DB at `vector_db/` is current

### Source path patch
Patch locations:
- `src/rag_loader.py` for fallback search PDF ingestion
- `src/rag_setup.py` for Chroma semantic PDF ingestion
- `src/config.py` only if base data root changes globally

## 10. Quick Demo Script

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Rebuild the RAG index:
   ```bash
   python -m src.rag_setup
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
4. Open the `RAG Assistant` page and ask a maintenance question.

---

### Appendix: Key functions

- `src.rag_utils.get_embedding(text)`
- `src.rag_setup.add_society_rules(collection, rules_path)`
- `src.rag_setup.add_service_logs(collection, logs_path)`
- `src.rag_setup.add_pdf_documents(collection, pdf_dirs)`
- `src.rag_loader.load_rag_data()`
- `src.rag_search.search_documents(question, top_k)`
- `src.rag_search._tfidf_search(question, top_k)`
