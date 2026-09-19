#!/usr/bin/env bash
# Render runs this at build time.
# Docs: https://render.com/docs/deploy-python

set -o errexit

pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# Download the embedding model now so it's cached in the build.
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
