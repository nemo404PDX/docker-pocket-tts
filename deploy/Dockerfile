FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
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
