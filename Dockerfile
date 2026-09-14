FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_HOST=0.0.0.0 \
    APP_PORT=8000 \
    HF_HOME=/models/huggingface \
    WESPEAKER_HOME=/models/wespeaker \
    DATABASE_PATH=/data/meeting-review.db

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgomp1 libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY static ./static
COPY run.py ./run.py

# faster-whisper 和 WeSpeaker 都会在首次使用时下载模型，模型目录必须挂载持久卷。
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /models/huggingface /models/wespeaker /data \
    && chown -R appuser:appuser /models/huggingface /models/wespeaker /data
VOLUME ["/models/huggingface", "/models/wespeaker", "/data"]

USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.getenv('APP_PORT','8000')+'/health', timeout=3)"

CMD ["python", "run.py"]
