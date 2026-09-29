FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY data/ ./data/

# logs/ lives outside the image so it persists across restarts when
# mounted as a volume (see docker-compose.yml). No database -- the
# catalog is loaded from data/deeplinks.json into memory at startup.
RUN mkdir -p /app/data /app/logs

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
