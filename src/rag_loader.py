import os
import pandas as pd
from pypdf import PdfReader
from src.config import DATA_DIR


def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        pages = [page.extract_text() for page in reader.pages]
        return "\n".join([p for p in pages if p]).strip()
    except Exception:
        return ""


def load_rag_data():

    documents = []

    try:
        rules_path = os.path.join(DATA_DIR, "rag", "society_rules.csv")
        rules_df = pd.read_csv(rules_path)
        for _, row in rules_df.iterrows():
            text = " ".join(row.astype(str))
            documents.append({"source": os.path.basename(rules_path), "text": text})
    except Exception:
        pass

    try:
        logs_path = os.path.join(DATA_DIR, "rag", "service_logs.csv")
        logs_df = pd.read_csv(logs_path)
        if "repair_notes" in logs_df.columns:
            for _, row in logs_df.iterrows():
                documents.append({"source": os.path.relpath(logs_path, DATA_DIR), "text": str(row["repair_notes"])})
    except Exception:
        pass

    pdf_dirs = [
        os.path.join(DATA_DIR, "rag"),
        os.path.join(DATA_DIR, "asset_manuals"),
        os.path.join(DATA_DIR, "rag", "asset_manuals")
    ]

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

                documents.append({
                    "source": os.path.relpath(pdf_path, DATA_DIR),
                    "text": text
                })

    return documents