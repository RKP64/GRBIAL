FROM python:3.12-slim
WORKDIR /srv
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

# Install CPU-only torch first to keep image small (~200MB vs ~2GB)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

# Download the embedding model at build time so cold starts don't re-download.
ENV HF_HOME=/srv/.cache/huggingface
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY app ./app
RUN useradd -m svc && mkdir -p /srv/data && chown -R svc /srv /srv/.cache
USER svc

# Render sets PORT dynamically. Fall back to 8000 for local dev.
ENV PORT=8000
EXPOSE ${PORT}

HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request,os;urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\",8000)}/healthz')"
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
