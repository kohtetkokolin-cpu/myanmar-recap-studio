FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    ffmpeg curl git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# yt-dlp breaks often (YouTube changes); nightly has the fastest fixes.
# Fall back to the stable release if the nightly install ever fails.
RUN pip install --no-cache-dir -U "yt-dlp[default]" && \
    (pip install --no-cache-dir -U --pre "yt-dlp[default]" || true)

# Pillow for thumbnail text overlay
RUN pip install --no-cache-dir Pillow

COPY . .

RUN mkdir -p downloads fonts cloned_voices data

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
