import os
import pandas as pd
import chromadb
from pypdf import PdfReader

from src.rag_utils import get_embedding
from src.config import BASE_DIR, DATA_DIR


def get_collection(client, name="amis_rag"):
    return client.get_or_create_collection(name=name)


def add_society_rules(collection, rules_path):
    if not os.path.exists(rules_path):
        return
    rules_df = pd.read_csv(rules_path)
    for i, row in rules_df.iterrows():
        text = " ".join(row.astype(str))
        if not text.strip():
            continue
        try:
            collection.add(
                ids=[f"rule_{i}"],
                documents=[text],
                embeddings=[get_embedding(text)],
                metadatas=[{"source": os.path.relpath(rules_path, BASE_DIR)}]
            )
        except Exception:
            continue


def add_service_logs(collection, logs_path):
    if not os.path.exists(logs_path):
        return
    logs_df = pd.read_csv(logs_path)
    if "repair_notes" not in logs_df.columns:
        return
    for i, row in logs_df.iterrows():
        repair_note = str(row.get("repair_notes", ""))
        if repair_note.strip() == "":
            continue
        try:
            collection.add(
                ids=[f"log_{i}"],
                documents=[repair_note],
                embeddings=[get_embedding(repair_note)],
                metadatas=[{"source": os.path.relpath(logs_path, BASE_DIR)}]
            )
        except Exception:
            continue


def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        pages = [page.extract_text() for page in reader.pages]
        text = "\n".join([p for p in pages if p])
        return text.strip()
    except Exception:
        return ""


def add_pdf_documents(collection, pdf_dirs):
    added = 0
    for pdf_dir in pdf_dirs:
        if not os.path.isdir(pdf_dir):
            continue

        for root, _, files in os.walk(pdf_dir):
            for filename in files:
                if not filename.lower().endswith(".pdf"):
                    continue

                pdf_path = os.path.join(root, filename)
                text = extract_text_from_pdf(pdf_path)
                if not text:
                    continue

                pdf_id = f"pdf_{os.path.relpath(pdf_path, BASE_DIR).replace(os.sep, '_')}"

                try:
                    collection.add(
                        ids=[pdf_id],
                        documents=[text],
                        embeddings=[get_embedding(text)],
                        metadatas=[{"source": os.path.relpath(pdf_path, BASE_DIR)}]
                    )
                    added += 1
                except Exception:
                    continue


def main():
    client = chromadb.PersistentClient(path=os.path.join(BASE_DIR, "vector_db"))

    existing_collections = [c.name for c in client.list_collections()]
    if "amis_rag" in existing_collections:
        client.delete_collection(name="amis_rag")
    collection = get_collection(client)
    rules_path = os.path.join(DATA_DIR, "rag", "society_rules.csv")
    logs_path = os.path.join(DATA_DIR, "rag", "service_logs.csv")
    pdf_dirs = [
        os.path.join(DATA_DIR, "rag"),
        os.path.join(DATA_DIR, "asset_manuals"),
        os.path.join(DATA_DIR, "rag", "asset_manuals")
    ]
    add_society_rules(collection, rules_path)
    add_service_logs(collection, logs_path)
    add_pdf_documents(collection, pdf_dirs)


if __name__ == "__main__":
    main()