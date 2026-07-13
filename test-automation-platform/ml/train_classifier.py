"""
Train a lightweight classifier for root-cause labeling using TF-IDF features.
Input: labeled CSV with columns: run_id, text_summary, root_cause_label
Saves model to models/root_cause_clf.joblib
"""
import os
import argparse
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def train(labeled_csv: str):
    df = pd.read_csv(labeled_csv, dtype=str).fillna("")
    texts = df["text_summary"].tolist()
    labels = df["root_cause_label"].tolist()
    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
    X = vectorizer.fit_transform(texts)
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    print(classification_report(y_test, preds))
    joblib.dump(vectorizer, os.path.join(MODELS_DIR, "clf_tfidf_vectorizer.joblib"))
    joblib.dump(clf, os.path.join(MODELS_DIR, "root_cause_clf.joblib"))
    print("Saved classifier and vectorizer to", MODELS_DIR)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="labeled CSV path")
    args = p.parse_args()
    train(args.csv)
