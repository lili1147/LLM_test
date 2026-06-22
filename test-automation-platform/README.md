# Test Automation Diagnostics Platform (Lightweight)

Overview:
- Lightweight platform to ingest historical failed test runs (CSV), build TF-IDF index,
  perform lightweight similarity search, train a root-cause classifier on TF-IDF features,
  and run a FastAPI diagnostics service that combines similarity + classifier + LLM suggestion.

Quickstart:
1. Install:
   pip install -r requirements.txt

2. Ingest historical CSV:
   python tools/ingest_csv.py --csv path/to/historical.csv

   historical.csv should contain columns: run_id, repo, commit, test_case, stacktrace, stdout, error_type

3. (Optional) Prepare labeled data for classifier:
   Create labeled CSV with columns: run_id, text_summary, root_cause_label
   python ml/train_classifier.py --csv labeled.csv

4. Run FastAPI service:
   uvicorn api.diagnostics_service:app --reload --port 8000
   (Set OPENAI_API_KEY in env if you want real LLM calls)

5. Call analyze:
   POST /analyze with JSON body match FailureEvent model.

Notes:
- Similarity uses TF-IDF cosine similarity for light-weight operation; no vector DB needed.
- LLM usage requires OPENAI_API_KEY; otherwise a stub response is returned.
- Remember to mask sensitive information before sending to cloud LLMs.
