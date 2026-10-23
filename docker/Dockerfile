# ---------------------------------------------------------------------------
# Stage 1: build the React frontend into static files
# ---------------------------------------------------------------------------
FROM node:22-slim AS frontend

WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
# FastAPI serves index.html at "/" and the assets under /static
RUN npm run build -- --base=/static/

# ---------------------------------------------------------------------------
# Stage 2: Python backend, serving the API + built frontend
# ---------------------------------------------------------------------------
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml ./
COPY *.py ./
RUN pip install --no-cache-dir ".[postgres]"

COPY --from=frontend /frontend/dist ./static

# Run as non-root; /app must be writable for the FAISS index and spec cache
RUN useradd --create-home app \
    && mkdir -p faiss_index \
    && chown -R app /app
USER app

EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
