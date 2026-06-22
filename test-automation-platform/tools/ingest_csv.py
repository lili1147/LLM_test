"""
Ingest historical CSV failures into TF-IDF index and metadata store.
Saves:
 - vectorizer -> models/tfidf_vectorizer.joblib
 - docs metadata -> data/historical_records.csv (normalized)
 - TF-IDF matrix (sparse) saved as joblib: models/tfidf_matrix.joblib
"""
import os
import re
import argparse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(__file__))  # project root

def mask_pii(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\b[0-9]{10,}\b", "<PHONE>", text)
    text = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "<EMAIL>", text)
    text = re.sub(r"(?i)token[:=]\s*[A-Za-z0-9\-_.]+", "TOKEN_REDACTED", text)
    return text

def summarize_row(row: dict) -> str:
    parts = []
    parts.append(str(row.get("repo", "")))
    parts.append(str(row.get("test_case", "")))
    parts.append(str(row.get("error_type", "")))
    # keep truncated stack/stdout to limit text length
    parts.append(str(row.get("stacktrace", ""))[:2000])
    parts.append(str(row.get("stdout", ""))[:1000])
    text = " ".join([p for p in parts if p])
    return mask_pii(text)

def ingest(csv_path: str, out_dir: str = None):
    if out_dir is None:
        out_dir = os.path.join(ROOT, "data")
    os.makedirs(out_dir, exist_ok=True)
    models_dir = os.path.join(ROOT, "models")
    os.makedirs(models_dir, exist_ok=True)

    df = pd.read_csv(csv_path, dtype=str).fillna("")
    docs = []
    ids = []
    metadatas = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        doc = summarize_row(row)
        docs.append(doc)
        ids.append(row.get("run_id") or str(_))
        metadatas.append({
            "run_id": row.get("run_id", ""),
            "repo": row.get("repo", ""),
            "test_case": row.get("test_case", ""),
            "commit": row.get("commit", ""),
            "error_type": row.get("error_type", "")
        })
    # TF-IDF
    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
    matrix = vectorizer.fit_transform(docs)
    # persist
    joblib.dump(vectorizer, os.path.join(models_dir, "tfidf_vectorizer.joblib"))
    joblib.dump(matrix, os.path.join(models_dir, "tfidf_matrix.joblib"))
    pd.DataFrame(metadatas).to_csv(os.path.join(out_dir, "historical_records.csv"), index=False)
    print("Ingested", len(docs), "records. Models saved to", models_dir)
    return True

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="path to historical csv")
    p.add_argument("--out", required=False, help="output data dir")
    args = p.parse_args()
    ingest(args.csv, args.out)
