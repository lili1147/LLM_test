"""
FastAPI service:
 - POST /analyze  : accept a failure event, return similarity results, classifier result, and LLM suggestion
Config via env:
 - OPENAI_API_KEY : if set, will call OpenAI ChatCompletion; otherwise returns stub suggestion
"""
import os
import json
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from services.similarity_search import query_similar
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict, Any
import openai
import requests

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_KEY:
    openai.api_key = OPENAI_KEY

ROOT = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(ROOT, "models")

# load classifier if exists
clf = None
clf_vectorizer = None
try:
    clf = joblib.load(os.path.join(MODELS_DIR, "root_cause_clf.joblib"))
    clf_vectorizer = joblib.load(os.path.join(MODELS_DIR, "clf_tfidf_vectorizer.joblib"))
except Exception:
    clf = None

app = FastAPI(title="Test Automation Diagnostics Service")

class FailureEvent(BaseModel):
    run_id: str
    repo: str
    commit: str = ""
    test_case: str = ""
    stacktrace: str = ""
    stdout: str = ""
    env: dict = {}

def build_summary(ev: FailureEvent) -> str:
    parts = [ev.repo, ev.test_case, ev.stacktrace[:2000], ev.stdout[:1000]]
    return " ".join([p for p in parts if p])

def call_llm_system(prompt: str) -> str:
    if OPENAI_KEY:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "你是资深自动化测试工程师，结合历史相似失败和上下文信息给出结构化诊断结果。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=800
        )
        return resp["choices"][0]["message"]["content"]
    else:
        # stubbed response for offline mode
        return json.dumps({
            "diagnosis": [{"cause":"AssertionError: expected X","confidence":0.8}],
            "reproduction_steps": ["Run pytest tests/test_login.py --maxfail=1"],
            "fix_suggestions": ["Adjust assertion or fix response payload"],
            "related": []
        })

@app.post("/analyze")
async def analyze(event: FailureEvent):
    summary = build_summary(event)
    # 1) similarity search
    similar = query_similar(summary, top_k=5)
    # 2) classifier prediction
    clf_result = None
    if clf is not None and clf_vectorizer is not None:
        x = clf_vectorizer.transform([summary])
        pred = clf.predict(x)[0]
        proba = max(clf.predict_proba(x)[0])
        clf_result = {"label": pred, "confidence": float(proba)}
    # 3) prepare LLM prompt: include top similar entries
    top_texts = "\n".join([f"- {s['repo']} {s['test_case']} (score={s['score']:.3f})" for s in similar])
    prompt = f"""失败摘要：
{summary}

分类器预测：{clf_result}
相似历史（top5）：
{top_texts}

请基于以上信息给出严格 JSON 格式的返回：
{{"diagnosis":[{{"cause":"...","confidence":0.0}}], "reproduction_steps":["..."], "fix_suggestions":["..."], "related":[{{"run_id":"...","note":"..."}}], "auto_issue_draft":{{"title":"...","body":"..."}}}}
"""
    llm_out = call_llm_system(prompt)
    # try parse JSON if possible
    parsed = None
    try:
        parsed = json.loads(llm_out)
    except Exception:
        parsed = {"raw": llm_out}
    return {"summary": summary[:1000], "similar": similar, "classifier": clf_result, "llm": parsed}
