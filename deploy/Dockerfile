FROM python:3.11-slim

WORKDIR /app

# libsndfile1 backs soundfile (wav/mp3/flac encoding). We deliberately do NOT install
# ffmpeg: torchaudio 2.x's CPU build only registers the `soundfile` audio backend, so an
# ffmpeg binary is never used for encoding — aac/opus stay unavailable (see supported_formats()).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements/ ./requirements/
RUN pip install --no-cache-dir -r requirements/cpu_requirements.txt

COPY app/ ./app/
COPY server.py ./
COPY deploy/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 49112

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["waitress-serve", "--host=0.0.0.0", "--port=49112", "app.wsgi:app"]
