FROM python:3.12-slim

WORKDIR /app

# System deps: none needed beyond what pip installs (sqlite3 is stdlib).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ensure the SQLite memory directory exists in the image (created fresh per deploy;
# Render's filesystem is ephemeral across deploys/restarts, which is expected for a demo).
RUN mkdir -p data/memory data/logs

EXPOSE 8000

# Render injects $PORT; dashboard_server.py reads it and binds 0.0.0.0 automatically.
CMD ["python", "dashboard_server.py"]
