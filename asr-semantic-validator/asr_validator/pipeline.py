"""
ASR semantic judgement pipeline:
 - normalize reference and hypothesis
 - compute WER
 - compute semantic similarity using sentence-transformers
 - decision rules:
    * if semantic_sim > accept_threshold -> ACCEPT
    * elif semantic_sim between mid_threshold and accept_threshold -> call LLM fallback
    * else -> REJECT
 - output JSON with verdict, confidence, tags, token-level diff (simple)
"""
import os
import json
from typing import Dict, Any
from .normalizer import normalize
from .metrics import ASRMetrics
import openai

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_KEY:
    openai.api_key = OPENAI_KEY

metrics = ASRMetrics()

ACCEPT_THRESHOLD = 0.92
MID_THRESHOLD = 0.80

def token_diff(ref: str, hyp: str):
    # simple token-level diff: returns list of substitutions/insertions/deletions
    r_tokens = ref.split()
    h_tokens = hyp.split()
    diffs = []
    # naive alignment: iterate min len and capture mismatches
    n = min(len(r_tokens), len(h_tokens))
    for i in range(n):
        if r_tokens[i] != h_tokens[i]:
            diffs.append({"ref": r_tokens[i], "asr": h_tokens[i], "type": "sub"})
    if len(r_tokens) > n:
        for t in r_tokens[n:]:
            diffs.append({"ref": t, "asr": "", "type": "del"})
    if len(h_tokens) > n:
        for t in h_tokens[n:]:
            diffs.append({"ref": "", "asr": t, "type": "ins"})
    return diffs

def call_llm_for_judgement(ref: str, hyp: str, context: str = "") -> Dict[str,Any]:
    prompt = {
        "system": "你是字幕质量评估专家。判断 ASR 输出是否在语义上与参考等价，返回 JSON.",
        "user": {
            "reference": ref,
            "asr": hyp,
            "context": context,
            "rules": "如果只是同义替换或语序变动但语义不变则 ACCEPT；数字/命名实体错误或信息缺失则 REJECT；给出标签和理由。"
        }
    }
    if OPENAI_KEY:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role":"system","content":prompt["system"]},
                {"role":"user","content":json.dumps(prompt["user"], ensure_ascii=False)}
            ],
            temperature=0.0,
            max_tokens=300
        )
        text = response["choices"][0]["message"]["content"]
        try:
            return json.loads(text)
        except Exception:
            return {"verdict":"UNKNOWN","confidence":0.0,"reason":text}
    else:
        # stubbed heuristic fallback
        return {"verdict":"ACCEPT_WITH_NOTE","confidence":0.8,"reason":"High semantic similarity fallback (stub)."}

def judge_segment(reference: str, hypothesis: str, context: str = "") -> Dict[str,Any]:
    ref_n = normalize(reference)
    hyp_n = normalize(hypothesis)
    w = metrics.compute_wer(ref_n, hyp_n)
    sim = metrics.semantic_similarity(ref_n, hyp_n)
    result = {"wer": w, "semantic_similarity": sim}
    if sim >= ACCEPT_THRESHOLD:
        result.update({"verdict":"ACCEPT","confidence":float(sim),"reason":"High semantic similarity","tags":["paraphrase"]})
    elif sim >= MID_THRESHOLD:
        # fallback to LLM
        llm_out = call_llm_for_judgement(ref_n, hyp_n, context)
        result.update({"verdict": llm_out.get("verdict"), "confidence": llm_out.get("confidence", 0.0), "reason": llm_out.get("reason",""), "tags": llm_out.get("tags",[])})
    else:
        result.update({"verdict":"REJECT","confidence":float(sim),"reason":"Low semantic similarity","tags":["low_similarity"]})
    result["token_level_diff"] = token_diff(ref_n, hyp_n)
    return result

if __name__ == "__main__":
    # demo
    ref = "把灯打开"
    hyp = "开灯"
    print(json.dumps(judge_segment(ref, hyp), ensure_ascii=False, indent=2))
