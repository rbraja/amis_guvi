# Apartment Maintenance Intelligence System (AMIS)

Lightweight ML + RAG toolkit for apartment maintenance analytics and assistant.

Quick start

- Install dependencies:

```bash
pip install -r requirements.txt
```

- Run the Streamlit app (pages auto-detected):

```bash
streamlit run app.py
```

- Build the vector database (RAG) from CSV sources:

```bash
python -m src.rag_setup
```

Project layout

- `app.py` - Streamlit app entry
- `pages/` - Streamlit multipage views
- `src/` - core modules (data loading, RAG utils, ML models)
- `data/` - raw, cleaned and RAG CSVs
- `vector_db/` - chroma DB files

Notes

- `src/rag_setup.py` creates/updates the Chroma collection used by the RAG assistant.
- `requirements.txt` lists the main packages; pin versions as needed for reproducible installs.

Contributions

Open an issue or submit a PR with improvements or bug fixes.
