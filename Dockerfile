# Serves the trained model as a REST API.
# Build AFTER running `python run_all.py` so outputs/models/pdm_bundle.joblib exists.
#   docker build -t pdm-api .
#   docker run -p 8000:8000 pdm-api      ->  http://localhost:8000/docs
FROM python:3.11-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

# The API needs no PyTorch: install only what serving requires
RUN pip install --no-cache-dir numpy pandas scikit-learn lightgbm shap joblib fastapi uvicorn pydantic

COPY config.py ./
COPY src/ ./src/
COPY api/ ./api/
COPY outputs/models/pdm_bundle.joblib ./outputs/models/pdm_bundle.joblib

EXPOSE 8000
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
