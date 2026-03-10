# mem0-api

Independent mem0 service for AI groupmate user-profile memory.

## Files

- `main.py`: FastAPI app with Bearer token auth and mem0 routes
- `Dockerfile`: image build
- `docker-compose.yml`: container deployment on the Milvus host
- `.env.example`: required environment variables

## Deploy

```bash
cp .env.example .env
docker compose up -d --build
```

The compose file expects the existing external Docker network named `milvus`.

## Endpoints

- `GET /healthz`
- `GET /readyz`
- `POST /memories`
- `POST /search`
- `GET /memories`
- `GET /memories/{memory_id}`
- `PUT /memories/{memory_id}`
- `GET /memories/{memory_id}/history`
- `DELETE /memories/{memory_id}`
- `DELETE /memories`
- `POST /reset`
