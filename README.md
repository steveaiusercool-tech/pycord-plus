# PyCord Plus

A free, Discord-style chat: FastAPI + WebSockets + SQLite, with reactions, edits, uploads, mentions, link previews, profile pics, and Tenor GIFs.

## Quick start

1) Python
- python -m venv .venv && source .venv/bin/activate
- pip install -r requirements.txt
- cp .env.example .env
- Set TENOR_API_KEY in .env (free API key)
- uvicorn server:app --reload
- open http://localhost:8000

2) Docker
- cp .env.example .env
- docker compose up --build
- open http://localhost:8000

## Notes
- Tenor API is free; get a key: https://developers.google.com/tenor/guides/quickstart
- Data stored locally in chat.db and uploads/ (bind mount in Docker)
- For production: use real auth (JWT), HTTPS, and rate limits
