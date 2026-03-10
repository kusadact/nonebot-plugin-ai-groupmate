import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from mem0 import Memory
from pydantic import BaseModel, Field

load_dotenv()


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


API_TOKEN = os.getenv("API_TOKEN", "").strip()
MILVUS_URL = _require_env("MILVUS_URL")
MILVUS_TOKEN = _require_env("MILVUS_TOKEN")
MILVUS_DB = os.getenv("MILVUS_DB", "default").strip() or "default"
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "user_profile").strip() or "user_profile"
MILVUS_METRIC = os.getenv("MILVUS_METRIC", "COSINE").strip() or "COSINE"
MEM0_EMBED_DIMS = int(os.getenv("MEM0_EMBED_DIMS", "1024"))
HISTORY_DB_PATH = os.getenv("HISTORY_DB_PATH", "/data/history.db").strip() or "/data/history.db"

LLM_BASE_URL = _require_env("LLM_BASE_URL")
LLM_API_KEY = _require_env("LLM_API_KEY")
LLM_MODEL = _require_env("LLM_MODEL")

EMBED_BASE_URL = _require_env("EMBED_BASE_URL")
EMBED_API_KEY = _require_env("EMBED_API_KEY")
EMBED_MODEL = _require_env("EMBED_MODEL")

Path(HISTORY_DB_PATH).parent.mkdir(parents=True, exist_ok=True)

MEMORY_CONFIG = {
    "version": "v1.1",
    "vector_store": {
        "provider": "milvus",
        "config": {
            "url": MILVUS_URL,
            "token": MILVUS_TOKEN,
            "collection_name": MILVUS_COLLECTION,
            "embedding_model_dims": MEM0_EMBED_DIMS,
            "metric_type": MILVUS_METRIC,
            "db_name": MILVUS_DB,
        },
    },
    "llm": {
        "provider": "openai",
        "config": {
            "api_key": LLM_API_KEY,
            "openai_base_url": LLM_BASE_URL,
            "model": LLM_MODEL,
            "temperature": 0.1,
        },
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "api_key": EMBED_API_KEY,
            "openai_base_url": EMBED_BASE_URL,
            "model": EMBED_MODEL,
            "embedding_dims": MEM0_EMBED_DIMS,
        },
    },
    "history_db_path": HISTORY_DB_PATH,
}

MEMORY_INSTANCE = Memory.from_config(MEMORY_CONFIG)

app = FastAPI(
    title="mem0-api",
    description="mem0 service for AI groupmate user profile memory.",
    version="1.0.0",
)


def require_api_token(authorization: str | None = Header(default=None)) -> None:
    if not API_TOKEN:
        return
    if not authorization:
        raise HTTPException(status_code=401, detail="missing Authorization header")

    token = authorization.strip()
    if token.lower().startswith("bearer "):
        token = token[7:].strip()

    if token != API_TOKEN:
        raise HTTPException(status_code=401, detail="invalid api token")


class Message(BaseModel):
    role: str = Field(..., description="Role of the message.")
    content: str = Field(..., description="Message content.")


class MemoryCreate(BaseModel):
    messages: list[Message] = Field(..., description="Messages to extract memories from.")
    user_id: str | None = None
    agent_id: str | None = None
    run_id: str | None = None
    metadata: dict[str, Any] | None = None


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query.")
    user_id: str | None = None
    run_id: str | None = None
    agent_id: str | None = None
    filters: dict[str, Any] | None = None
    limit: int | None = 5


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/healthz", summary="Liveness probe")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz", summary="Readiness probe")
def readyz() -> dict[str, Any]:
    collections = MEMORY_INSTANCE.vector_store.client.list_collections()
    return {
        "status": "ready",
        "database": MILVUS_DB,
        "collection": MILVUS_COLLECTION,
        "collection_exists": MILVUS_COLLECTION in collections,
    }


@app.post("/memories", summary="Create memories", dependencies=[Depends(require_api_token)])
def add_memory(memory_create: MemoryCreate):
    if not any([memory_create.user_id, memory_create.agent_id, memory_create.run_id]):
        raise HTTPException(status_code=400, detail="At least one identifier is required.")

    params = {
        key: value
        for key, value in memory_create.model_dump().items()
        if value is not None and key != "messages"
    }
    try:
        response = MEMORY_INSTANCE.add(
            messages=[message.model_dump() for message in memory_create.messages],
            **params,
        )
        return JSONResponse(content=response)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/memories", summary="Get memories", dependencies=[Depends(require_api_token)])
def get_all_memories(
    user_id: str | None = None,
    run_id: str | None = None,
    agent_id: str | None = None,
    limit: int = 100,
):
    if not any([user_id, run_id, agent_id]):
        raise HTTPException(status_code=400, detail="At least one identifier is required.")

    params = {
        key: value
        for key, value in {"user_id": user_id, "run_id": run_id, "agent_id": agent_id, "limit": limit}.items()
        if value is not None
    }
    try:
        return MEMORY_INSTANCE.get_all(**params)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/memories/{memory_id}", summary="Get a memory", dependencies=[Depends(require_api_token)])
def get_memory(memory_id: str):
    try:
        return MEMORY_INSTANCE.get(memory_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/search", summary="Search memories", dependencies=[Depends(require_api_token)])
def search_memories(search_req: SearchRequest):
    try:
        params = {
            key: value
            for key, value in search_req.model_dump().items()
            if value is not None and key != "query"
        }
        return MEMORY_INSTANCE.search(query=search_req.query, **params)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.put("/memories/{memory_id}", summary="Update a memory", dependencies=[Depends(require_api_token)])
def update_memory(memory_id: str, updated_memory: dict[str, Any]):
    try:
        return MEMORY_INSTANCE.update(memory_id=memory_id, data=updated_memory)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/memories/{memory_id}/history", summary="Get memory history", dependencies=[Depends(require_api_token)])
def memory_history(memory_id: str):
    try:
        return MEMORY_INSTANCE.history(memory_id=memory_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.delete("/memories/{memory_id}", summary="Delete a memory", dependencies=[Depends(require_api_token)])
def delete_memory(memory_id: str):
    try:
        MEMORY_INSTANCE.delete(memory_id=memory_id)
        return {"message": "Memory deleted successfully"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.delete("/memories", summary="Delete all memories", dependencies=[Depends(require_api_token)])
def delete_all_memories(
    user_id: str | None = None,
    run_id: str | None = None,
    agent_id: str | None = None,
):
    if not any([user_id, run_id, agent_id]):
        raise HTTPException(status_code=400, detail="At least one identifier is required.")

    params = {
        key: value
        for key, value in {"user_id": user_id, "run_id": run_id, "agent_id": agent_id}.items()
        if value is not None
    }
    try:
        MEMORY_INSTANCE.delete_all(**params)
        return {"message": "All relevant memories deleted"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/reset", summary="Reset all memories", dependencies=[Depends(require_api_token)])
def reset_memory():
    try:
        MEMORY_INSTANCE.reset()
        return {"message": "All memories reset"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
