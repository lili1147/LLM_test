"""
WER calculation (jiwer) and semantic similarity via sentence-transformers.
"""
from jiwer import wer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

class ASRMetrics:
    def __init__(self, model_name=MODEL_NAME):
        self.embed_model = SentenceTransformer(model_name)

    def compute_wer(self, ref: str, hyp: str) -> float:
        try:
            return wer(ref, hyp)
        except Exception:
            return 1.0

    def semantic_similarity(self, ref: str, hyp: str) -> float:
        emb = self.embed_model.encode([ref, hyp], convert_to_numpy=True)
        sim = cosine_similarity([emb[0]], [emb[1]])[0][0]
        return float(sim)
