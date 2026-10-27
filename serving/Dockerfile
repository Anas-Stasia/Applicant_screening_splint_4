# syntax=docker/dockerfile:1
# Applicant Screening API — production CPU image (Sprint 4)
#
# Build from the applicant_screening_nb/ folder (the repo root):
#   docker build -f serving/Dockerfile -t screening-api --build-arg GIT_SHA=$(git rev-parse --short HEAD) .
#   docker run -p 8000:8000 -v screening-logs:/srv/logs screening-api
# or simply:  docker compose up --build
#
# Works on Render, Railway, AWS ECS and Fly unchanged: the port comes from $PORT.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models

WORKDIR /srv

# 1. Dependencies (cached layer — only rebuilt when requirements change).
#    CPU-only torch keeps the image ~1 GB instead of ~5 GB with CUDA.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY serving/requirements-core.txt serving/requirements.txt ./
RUN pip install -r requirements.txt

# 2. Bake the model into the image so containers start without internet access
#    and every replica runs byte-identical weights.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# 3. App code + the committed artifact bundle (serving/artifacts, see
#    scripts/export_artifacts.py). A fresh git clone builds — no notebooks needed.
COPY serving/app ./app
COPY serving/demo ./demo
COPY serving/scripts ./scripts
COPY serving/artifacts /data

# 4. Run as an unprivileged user; only the log folder is writable.
RUN useradd --create-home --uid 10001 app && mkdir -p /srv/logs && chown app:app /srv/logs
USER app

ARG GIT_SHA=dev
ENV SCREENING_ARTIFACTS_DIR=/data \
    SCREENING_LOG_DIR=/srv/logs \
    SCREENING_LOG_FORMAT=json \
    SCREENING_ENVIRONMENT=docker \
    SCREENING_GIT_SHA=${GIT_SHA} \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PORT=8000

EXPOSE 8000
VOLUME ["/srv/logs"]
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8000\")}/ready', timeout=4)"

# One worker per container: each worker holds its own copy of the model (~400 MB),
# so scale out with more containers. WEB_CONCURRENCY overrides it if RAM allows.
# --proxy-headers: Render/Railway/ALB terminate TLS in front of us.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers ${WEB_CONCURRENCY:-1} --proxy-headers --forwarded-allow-ips='*'"]
