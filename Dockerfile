FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY pyproject.toml Makefile ./
COPY src ./src
COPY data/postings.csv ./data/postings.csv
ENV JOBREC_EMBEDDER=tfidf JOBREC_LLM_PROVIDER=none PYTHONPATH=/app/src
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "jobrec.api:app", "--host", "0.0.0.0", "--port", "8000"]
