# ASR Semantic Validator

Overview:
- Pipeline to judge ASR subtitle segments with semantic-aware decisions.
- Combines normalization, WER, semantic similarity (sentence-transformers), and optional LLM fallback.

Quickstart:
1. Install:
   pip install -r requirements.txt

2. Run demo:
   python -c "from asr_validator.pipeline import judge_segment; import json; print(json.dumps(judge_segment('把灯打开','开灯'), ensure_ascii=False, indent=2))"

3. For bulk evaluation:
   - Read sample_data/sample.jsonl, iterate segments, call judge_segment for each,
     collect AI judgments and send low-confidence ones to manual review UI (schema in ui/review_schema.json).

Notes:
- Use OPENAI_API_KEY env var for real LLM fallback.
- Tune ACCEPT_THRESHOLD and MID_THRESHOLD in pipeline.py per your dataset.
- Save manual review labels back to dataset for active learning and threshold tuning.
