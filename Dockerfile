FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and pipeline artifacts
COPY src/ ./src/
COPY model.pkl preprocessor.pkl metrics.json dataset_summary.json ./

# Copy MLflow tracking store so the dashboard has run history
COPY mlruns/ ./mlruns/

# The app reads params.yaml for monitoring thresholds
COPY params.yaml .

# predictions.jsonl is created at runtime; no COPY needed
# Create it as a writable empty file so the container starts cleanly
RUN touch predictions.jsonl

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
