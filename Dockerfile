FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_HOST=0.0.0.0 \
    APP_PORT=8000 \
    HF_HOME=/models/huggingface \
    DATABASE_PATH=/data/meeting-review.db

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY static ./static
COPY run.py ./run.py

# faster-whisper 会在首次转写时下载模型。生产环境请把此目录挂载为持久卷，
# 或将预下载的模型目录通过 WHISPER_MODEL 指向挂载路径。
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /models/huggingface /data \
    && chown -R appuser:appuser /models/huggingface /data
VOLUME ["/models/huggingface", "/data"]

USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.getenv('APP_PORT','8000')+'/health', timeout=3)"

CMD ["python", "run.py"]
