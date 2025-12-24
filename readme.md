# Eve Online — PgVector + LangGraph RAG Demo

This repository is a small technology demonstrator that shows how to build a Resource-Augmented Generation (RAG) pipeline using:

- PgVector (Postgres extension) as the vector database
- Ollama as the Local LLM provider (embeddings + LLMs)
- LangGraph to orchestrate the agent workflow (used inside the bot)
- A simple FastAPI backend exposing an agent API
- A Streamlit frontend for demo interaction

Important: This project is a demonstrator and is NOT suitable for production use — there is no authentication/authorization, the agent is a singleton, and there are minimal operational safeguards.

This setup assumes that you are doing this on Windows using WSL. See environment setup instructions here https://learn.microsoft.com/en-us/windows/python/web-frameworks



**Repository layout**

- `compose.yml` — Docker Compose services used for the demo (Postgres+pgvector, Ollama, bot, frontend)
- `Dockerfile` / `Dockerfile.frontend` — container images for backend and frontend
- `src/evewikicrawler/crawler.py` — one-off script that crawls MediaWiki and writes embeddings to PgVector
- `src/evewikibot/` — FastAPI backend, bot service and LangGraph orchestration code
- `src/evewikibot/sql/initialize_db.sql` — SQL to create the vector table

**Quick summary**

- Start the demo stack with Docker Compose (Postgres with pgvector, Ollama, backend and frontend).
- Run the MediaWiki crawler once to populate the PgVector collection with page chunks and embeddings.
- Use the Streamlit frontend to interact with the agent which performs semantic search against PgVector and uses Ollama for LLM reasoning.

**Prerequisites**

- Docker and Docker Compose (modern `docker compose` CLI)
- Python 3.11+ for running the crawler locally
- (Optional) A local Python virtual environment for running the crawler

**Environment**

Create a `.env` file at the repository root with the variables used by `compose.yml` and the services. Example values (tweak for your environment):

```env
# Postgres
POSTGRES_DB=evewiki
POSTGRES_USER=[your_postgres_username]
POSTGRES_PASSWORD=[your_postgres_password]

# Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma3:1b
OLLAMA_EMBEDDING_MODEL=embeddinggemma:300m
OLLAMA_TEMPERATURE=0.7

# PgVector client settings used by the backend and crawler
PGVECTOR_HOST=pgvector
PGVECTOR_PORT=5432
PGVECTOR_DATABASE=evewiki
PGVECTOR_USER=[your_postgres_username]
PGVECTOR_PASSWORD=[your_postgres_password]
PGVECTOR_COLLECTION=eve_wiki
PGVECTOR_EMBEDDING_DIM=384
PGVECTOR_TOP_K=10

# Debug
DEBUG=true

# Crawler
EVEWIKI_API_URL=https://wiki.eveuniversity.org/api.php
CRAWLER_RATE_LIMIT_QPM=30
CRAWLER_CHUNK_SIZE=1000
CRAWLER_CHUNK_OVERLAP=200
CRAWLER_BATCH_SIZE=10
```

**Initialize and run the demo (recommended flow)**

1. Start services with Docker Compose

```bash
docker compose --env-file .env up --build
```

This brings up:

- `pgvector` (Postgres with pgvector extension) on port 5432
- `ollama` LLM service on port 11434
- `evewikibot` backend (internal service)
- `evewikibotfront` Streamlit frontend on port 8501

2. Initialize the database table for vectors

The SQL used to create the vector table is in `src/evewikibot/sql/initialize_db.sql`.

From the host you can apply it with `psql` (example):

```bash
# Option A: run psql from host against the DB container
psql "host=localhost port=5432 dbname=${POSTGRES_DB} user=${POSTGRES_USER} password=${POSTGRES_PASSWORD}" -f src/evewikibot/sql/initialize_db.sql

# Option B: pipe the file into the DB container's psql
docker exec -i eve_pgvector psql -U ${POSTGRES_USER} -d ${POSTGRES_DB} < src/evewikibot/sql/initialize_db.sql
```

3. Populate PgVector via the crawler (run once)

The crawler is a standalone script that fetches pages from the configured MediaWiki API, splits them into chunks, generates embeddings via Ollama, and stores them in PgVector.

Run locally (recommended):

```bash
source .venv/bin/activate
pip install -r requirements.txt

# Ensure the .env is in the repo root and services (Postgres + Ollama) are running
python src/evewikicrawler/crawler.py
```

Notes:

- The crawler respects `EVEWIKI_API_URL` and a rate limit (`CRAWLER_RATE_LIMIT_QPM`) to avoid overloading the wiki.
- Embeddings are requested from Ollama at `OLLAMA_BASE_URL` using the model in `OLLAMA_EMBEDDING_MODEL`.

4. Use the demo frontend

Open the Streamlit UI at http://localhost:8501. The frontend connects to the backend service inside the Docker Compose network.

If you prefer to call the backend directly (for development), run the backend locally with the src/ path on `PYTHONPATH`:

```bash
# Start backend locally (development)
PYTHONPATH=src uvicorn evewikibot.main:app --host 0.0.0.0 --port 8000 --reload
```

Then you can access the API at http://localhost:8000 and the OpenAPI docs at http://localhost:8000/docs.

**Important files**

- SQL schema: [src/evewikibot/sql/initialize_db.sql](src/evewikibot/sql/initialize_db.sql)
- Backend entrypoint: [src/evewikibot/main.py](src/evewikibot/main.py)
- Crawler: [src/evewikicrawler/crawler.py](src/evewikicrawler/crawler.py)

**Troubleshooting**

- If the crawler fails to connect to the DB, verify `PGVECTOR_HOST`/`PGVECTOR_PORT` and credentials in `.env` and that the `pgvector` container is healthy.
- If Ollama embedding requests fail, check the `OLLAMA_BASE_URL` and that the Ollama container has the required models pulled. The `ollama` service in `compose.yml` includes post-start commands to pull recommended models.
- If the frontend cannot load results, open the browser console and check that the frontend container can reach the backend (when using Docker Compose the service names are resolvable inside the network).

**Security & Production notes**

- This repository is a learning/demo project. It intentionally omits production readiness features: no authentication/authorization, no request quotaing, no secrets management, and the bot runs as a singleton.
- Do not expose this stack to untrusted networks without adding proper security layers.

# Notes on quality

Currently this agent is pretty stupid and cannot give out proper answers that are not hallucinations or just plain wrong. This is just a way to show how to chunk sources for RAG, expand user queries for semantic search, and provide answers based on information. To improve results consider following upgrades:
- Use searched chunks to fetch whole webpage for better context
- Use better LLM models. Currently this project uses Gemma3:1b (container also fetches Qwen3:1.7b for testing). It is very limited model that can be run on a laptop.
- Query expansion can give out pretty wild results occasionally. This can be improved with more precise instructions.
