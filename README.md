# HMS AI Service

Internal AI/ML microservice for the HMS platform.

## Tech Stack

- **Framework:** FastAPI
- **Runtime:** Python 3.11
- **Server:** Uvicorn
- **Validation:** Pydantic V2

## Endpoints

| Method | Path       | Description                          |
|--------|------------|--------------------------------------|
| GET    | `/health`  | Health check, returns service status |
| POST   | `/predict` | Dummy prediction (V1 placeholder)    |

## Local Development

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the server:

```bash
uvicorn app.main:app --reload
```

The service runs on port 8000 by default. API docs are available at `http://localhost:8000/docs`.

## Docker

Build and run standalone:

```bash
docker build -t hms-ai .
docker run -p 8000:8000 hms-ai
```

Typically this service is run via `docker compose` from the `hms-infra` repository rather than standalone.

## Architecture Notes

- This is an **internal service** and is not publicly exposed through nginx.
- The backend calls it via HTTP when AI/ML predictions are needed.
- Port 8000 is only accessible within the Docker network.
