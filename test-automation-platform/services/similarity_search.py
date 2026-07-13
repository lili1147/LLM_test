"""
Lightweight TF-IDF similarity search using saved vectorizer + matrix.
Provides function query_similar(text, top_k) -> list of metadata + score
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

ROOT = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(ROOT, "models")
DATA_DIR = os.path.join(ROOT, "data")

def load_index():
    vectorizer = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))
    tfidf_matrix = joblib.load(os.path.join(MODELS_DIR, "tfidf_matrix.joblib"))
    meta = pd.read_csv(os.path.join(DATA_DIR, "historical_records.csv"), dtype=str).fillna("")
    return vectorizer, tfidf_matrix, meta

def query_similar(text: str, top_k: int = 5):
    vectorizer, matrix, meta = load_index()
    qv = vectorizer.transform([text])
    # compute cosine similarity with stored matrix
    sims = cosine_similarity(qv, matrix).flatten()  # shape (n_records,)
    idx = np.argsort(-sims)[:top_k]
    results = []
    for i in idx:
        results.append({
            "run_id": meta.iloc[i]["run_id"],
            "repo": meta.iloc[i]["repo"],
            "test_case": meta.iloc[i]["test_case"],
            "commit": meta.iloc[i]["commit"] if "commit" in meta.columns else "",
            "error_type": meta.iloc[i].get("error_type",""),
            "score": float(sims[i])
        })
    return results

if __name__ == "__main__":
    sample = "Traceback: AssertionError in tests/test_login.py expected 200 got 500"
    print(query_similar(sample, top_k=3))
