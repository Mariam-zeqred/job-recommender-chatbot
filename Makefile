.PHONY: install check lint test data vectors eval serve clean

PY ?= python3

install:
	$(PY) -m pip install -r requirements.txt

lint:
	$(PY) -m ruff check src tests scripts
	$(PY) -m ruff format --check src tests scripts

test:
	$(PY) -m pytest -q

# Regenerate the synthetic postings (deterministic: same seed -> byte-identical CSV).
data:
	$(PY) scripts/generate_postings.py

# Optional: GloVe 50-d word vectors (66 MB, public domain) for the "glove" embedder.
vectors:
	$(PY) scripts/fetch_vectors.py

# Precision@k of each embedder on data/eval (writes figures/eval_results.json).
eval:
	$(PY) scripts/evaluate.py

serve:
	$(PY) -m uvicorn jobrec.api:app --app-dir src --reload --port 8000

# "make check" is the smoke test: lint + unit tests + the data generator is reproducible + eval runs.
check: lint test
	$(PY) scripts/generate_postings.py --check
	$(PY) scripts/evaluate.py --quick

clean:
	rm -rf .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -exec rm -rf {} +
