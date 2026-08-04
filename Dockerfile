FROM python:3.12-slim
WORKDIR /srv
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

COPY requirements.txt .
RUN pip install -r requirements.txt

# Download the embedding model at build time so cold starts don't wait 30s.
# The model is cached under /srv/.cache and reused on every container start.
ENV HF_HOME=/srv/.cache/huggingface
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY app ./app
RUN useradd -m svc && mkdir -p /srv/data && chown -R svc /srv /srv/.cache
USER svc
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/healthz')"
CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8000"]
