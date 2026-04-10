#!/usr/bin/env python3
"""
Rebuild Qdrant vectors from SQL.

Design goals:
1. Rebuild chat vectors from SQL chat history by regrouping messages into contexts.
2. Rebuild media vectors from SQL media rows and local image files.
3. Default media filter is conservative: only rebuild rows that are already vectorized,
   not blocked, and whose files still exist.
4. Optionally reset target collections before rebuilding.

Run this script in the same Python environment as the plugin.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import socket
import sys
import time
import hashlib
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient, models
from sqlalchemy import bindparam, create_engine, text

try:
    import tiktoken
except Exception:
    tiktoken = None


CHAT_TABLE = "nonebot_plugin_ai_groupmate_chathistory"
MEDIA_TABLE = "nonebot_plugin_ai_groupmate_mediastorage"
CHAT_COLLECTION = "chat_collection"
MEDIA_COLLECTION = "media_collection"
DEFAULT_CONFIG_PATH = Path(__file__).with_name("rebuild_qdrant_vectors.config")
DASHSCOPE_MULTIMODAL_BATCH_LIMIT = 5
HTTP_RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, os.getenv(name.upper(), default))


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _load_config_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    config: dict[str, str] = {}
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"invalid config line {line_no}: {raw_line}")
        key, value = line.split("=", 1)
        key = key.strip().upper()
        if not key:
            raise ValueError(f"invalid config line {line_no}: empty key")
        config[key] = _strip_quotes(value)
    return config


def _preparse_config_path(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    known, _ = parser.parse_known_args(argv)
    return Path(known.config).expanduser()


def _cfg(config: dict[str, str], key: str, env_value: str = "", default: str = "") -> str:
    value = config.get(key.upper(), "")
    if value:
        return value
    if env_value:
        return env_value
    return default


def _cfg_int(config: dict[str, str], key: str, env_value: str = "", default: int = 0) -> int:
    raw = _cfg(config, key, env_value, str(default))
    return int(raw or default)


def _cfg_bool(config: dict[str, str], key: str, env_value: str = "", default: bool = False) -> bool:
    raw = _cfg(config, key, env_value, "true" if default else "false").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _normalize_db_url(raw_url: str) -> str:
    replacements = {
        "sqlite+aiosqlite://": "sqlite://",
        "postgresql+asyncpg://": "postgresql+psycopg://",
        "mysql+aiomysql://": "mysql+pymysql://",
    }
    for src, dst in replacements.items():
        if raw_url.startswith(src):
            return raw_url.replace(src, dst, 1)
    return raw_url


def _estimate_token_count(text_value: str) -> int:
    if not text_value:
        return 0
    if tiktoken is None:
        return len(text_value)
    try:
        encoder = tiktoken.get_encoding("cl100k_base")
        return len(encoder.encode(text_value))
    except Exception:
        return len(text_value)


def _image_to_base64(image_path: str) -> str:
    if image_path.startswith("data:image/"):
        _, _, payload = image_path.partition(",")
        return payload
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


@dataclass
class ChatRow:
    msg_id: int
    session_id: str
    user_name: str
    content: str
    created_at: datetime


@dataclass
class MediaRow:
    media_id: int
    file_path: str
    description: str
    blocked: bool
    vectorized: bool
    references: int
    created_at: datetime


class RemoteModelClient:
    def __init__(
        self,
        base_url: str = "",
        api_key: str = "",
        embedding_base_url: str = "",
        embedding_api_key: str = "",
        embedding_model: str = "",
        embedding_dimensions: int = 0,
        media_embedding_provider: str = "openai",
        media_embedding_base_url: str = "",
        media_embedding_api_key: str = "",
        media_embedding_model: str = "",
        media_embedding_dimensions: int = 0,
        clip_base_url: str = "",
        clip_api_key: str = "",
        http_max_retries: int = 5,
        http_retry_sleep_seconds: float = 2.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.embedding_base_url = embedding_base_url.rstrip("/")
        self.embedding_api_key = embedding_api_key
        self.embedding_model = embedding_model
        self.embedding_dimensions = embedding_dimensions
        self.media_embedding_provider = (media_embedding_provider or "openai").strip().lower()
        self.media_embedding_base_url = media_embedding_base_url.rstrip("/")
        self.media_embedding_api_key = media_embedding_api_key
        self.media_embedding_model = media_embedding_model
        self.media_embedding_dimensions = media_embedding_dimensions
        self.clip_base_url = clip_base_url.rstrip("/")
        self.clip_api_key = clip_api_key
        self.http_max_retries = max(1, http_max_retries)
        self.http_retry_sleep_seconds = max(0.5, http_retry_sleep_seconds)

    def _post_json_with_base(self, base_url: str, path: str, payload: dict[str, Any], api_key: str = "") -> dict[str, Any]:
        if not base_url:
            raise RuntimeError("base_url is empty")
        url = f"{base_url}{path}"
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        data = json.dumps(payload).encode("utf-8")
        last_error: Exception | None = None
        body = ""
        for attempt in range(1, self.http_max_retries + 1):
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    body = resp.read().decode("utf-8")
                break
            except urllib.error.HTTPError as e:
                detail = ""
                try:
                    detail = e.read().decode("utf-8")
                except Exception:
                    pass
                retryable = e.code in HTTP_RETRYABLE_STATUS_CODES
                last_error = RuntimeError(f"http {e.code}: {detail or e.reason}")
                print(
                    f"[http] request failed attempt={attempt}/{self.http_max_retries} "
                    f"status={e.code} retryable={retryable} url={url} detail={detail or e.reason}"
                )
                if not retryable or attempt >= self.http_max_retries:
                    raise last_error from e
            except (urllib.error.URLError, TimeoutError, socket.timeout) as e:
                last_error = RuntimeError(f"request failed: {e}")
                print(
                    f"[http] request failed attempt={attempt}/{self.http_max_retries} "
                    f"retryable=true url={url} detail={e}"
                )
                if attempt >= self.http_max_retries:
                    raise last_error from e
            time.sleep(self.http_retry_sleep_seconds * attempt)
        else:
            assert last_error is not None
            raise last_error
        return json.loads(body)

    @staticmethod
    def _normalize_embeddings_response(data: dict[str, Any]) -> list[list[float]]:
        dense: list[list[float]] = []
        for item in data.get("data", []):
            if isinstance(item, dict) and "embedding" in item:
                dense.append(item["embedding"])
        if not dense:
            raise RuntimeError(f"invalid embeddings response: {data}")
        return dense

    @staticmethod
    def _normalize_dashscope_embeddings_response(data: dict[str, Any]) -> list[list[float]]:
        dense: list[list[float]] = []
        output = data.get("output", {})
        for item in output.get("embeddings", []):
            if isinstance(item, dict) and "embedding" in item:
                dense.append(item["embedding"])
        if not dense:
            raise RuntimeError(f"invalid dashscope embeddings response: {data}")
        return dense

    def _post_embeddings(self, base_url: str, api_key: str, model: str, inputs: list[Any], dimensions: int = 0) -> list[list[float]]:
        if not model:
            raise RuntimeError("embedding model is empty")
        payload: dict[str, Any] = {"model": model, "input": inputs}
        if dimensions > 0:
            payload["dimensions"] = dimensions
        data = self._post_json_with_base(base_url, "/embeddings", payload, api_key)
        return self._normalize_embeddings_response(data)

    @staticmethod
    def _to_dashscope_image(image_input: str) -> str:
        if image_input.startswith("data:image/"):
            return image_input
        if image_input.startswith("http://") or image_input.startswith("https://"):
            return image_input
        if os.path.exists(image_input):
            mime_type = mimetypes.guess_type(image_input)[0] or "image/png"
            if not mime_type.startswith("image/"):
                mime_type = "image/png"
            with open(image_input, "rb") as image_file:
                payload = base64.b64encode(image_file.read()).decode("utf-8")
            return f"data:{mime_type};base64,{payload}"
        raise RuntimeError(f"unsupported image source: {image_input}")

    def _post_dashscope_media_embeddings(self, images: list[str]) -> list[list[float]]:
        if not self.media_embedding_model:
            raise RuntimeError("media embedding model is empty")

        dense: list[list[float]] = []
        for start in range(0, len(images), DASHSCOPE_MULTIMODAL_BATCH_LIMIT):
            items = [{"image": self._to_dashscope_image(item)} for item in images[start : start + DASHSCOPE_MULTIMODAL_BATCH_LIMIT]]
            payload: dict[str, Any] = {
                "model": self.media_embedding_model,
                "input": {"contents": items},
            }
            if self.media_embedding_dimensions > 0:
                payload["parameters"] = {"dimension": self.media_embedding_dimensions}
            data = self._post_json_with_base(
                self.media_embedding_base_url,
                "",
                payload,
                self.media_embedding_api_key or self.api_key,
            )
            dense.extend(self._normalize_dashscope_embeddings_response(data))
        return dense

    def _post_clip_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        clip_base = self.clip_base_url or self.base_url
        clip_key = self.clip_api_key or self.api_key
        if not clip_base:
            raise RuntimeError("clip base_url is empty")
        return self._post_json_with_base(clip_base, path, payload, clip_key)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self.embedding_base_url:
            return self._post_embeddings(
                self.embedding_base_url,
                self.embedding_api_key or self.api_key,
                self.embedding_model,
                texts,
                self.embedding_dimensions,
            )
        if self.base_url:
            data = self._post_json_with_base(self.base_url, "/embed", {"texts": texts}, self.api_key)
            return data.get("dense", [])
        raise RuntimeError("text embedding endpoint is not configured")

    def embed_media_images(self, image_paths: list[str]) -> list[list[float]]:
        if self.media_embedding_base_url:
            if self.media_embedding_provider == "aliyun_dashscope":
                return self._post_dashscope_media_embeddings(image_paths)
            inputs = [
                [{"type": "input_image", "image_url": f"data:image/png;base64,{_image_to_base64(path)}"}]
                for path in image_paths
            ]
            return self._post_embeddings(
                self.media_embedding_base_url,
                self.media_embedding_api_key or self.api_key,
                self.media_embedding_model,
                inputs,
                self.media_embedding_dimensions,
            )
        data = self._post_clip_json("/clip/image", {"images_base64": [_image_to_base64(path) for path in image_paths]})
        return data.get("dense", [])


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    config_path = _preparse_config_path(argv)
    config = _load_config_file(config_path)

    parser = argparse.ArgumentParser(description="Rebuild Qdrant vectors from SQL")
    parser.add_argument("--config", default=str(config_path), help="key=value config file path")
    parser.add_argument("--db-url", default=_cfg(config, "DB_URL", _env("DB_URL", _env("DATABASE_URL", ""))))
    parser.add_argument("--pic-dir", default=_cfg(config, "PIC_DIR", _env("PIC_DIR", "")))
    parser.add_argument("--qdrant-uri", default=_cfg(config, "QDRANT_URI", _env("ai_groupmate__qdrant_uri", "")))
    parser.add_argument("--qdrant-api-key", default=_cfg(config, "QDRANT_API_KEY", _env("ai_groupmate__qdrant_api_key", "")))
    parser.add_argument("--chat-vector-dim", type=int, default=_cfg_int(config, "CHAT_VECTOR_DIM", _env("ai_groupmate__chat_vector_dim", ""), 1024))
    parser.add_argument("--media-vector-dim", type=int, default=_cfg_int(config, "MEDIA_VECTOR_DIM", _env("ai_groupmate__media_vector_dim", ""), 2560))
    parser.add_argument("--chat-collection", default=_cfg(config, "CHAT_COLLECTION", default=CHAT_COLLECTION))
    parser.add_argument("--media-collection", default=_cfg(config, "MEDIA_COLLECTION", default=MEDIA_COLLECTION))
    parser.add_argument("--remote-model-base-url", default=_cfg(config, "REMOTE_MODEL_BASE_URL", _env("ai_groupmate__remote_model_base_url", "")))
    parser.add_argument("--remote-model-api-key", default=_cfg(config, "REMOTE_MODEL_API_KEY", _env("ai_groupmate__remote_model_api_key", "")))
    parser.add_argument("--remote-embedding-base-url", default=_cfg(config, "REMOTE_EMBEDDING_BASE_URL", _env("ai_groupmate__remote_embedding_base_url", "")))
    parser.add_argument("--remote-embedding-api-key", default=_cfg(config, "REMOTE_EMBEDDING_API_KEY", _env("ai_groupmate__remote_embedding_api_key", "")))
    parser.add_argument("--remote-embedding-model", default=_cfg(config, "REMOTE_EMBEDDING_MODEL", _env("ai_groupmate__remote_embedding_model", "")))
    parser.add_argument("--remote-media-embedding-provider", default=_cfg(config, "REMOTE_MEDIA_EMBEDDING_PROVIDER", _env("ai_groupmate__remote_media_embedding_provider", "aliyun_dashscope")))
    parser.add_argument("--remote-media-embedding-base-url", default=_cfg(config, "REMOTE_MEDIA_EMBEDDING_BASE_URL", _env("ai_groupmate__remote_media_embedding_base_url", "")))
    parser.add_argument("--remote-media-embedding-api-key", default=_cfg(config, "REMOTE_MEDIA_EMBEDDING_API_KEY", _env("ai_groupmate__remote_media_embedding_api_key", "")))
    parser.add_argument("--remote-media-embedding-model", default=_cfg(config, "REMOTE_MEDIA_EMBEDDING_MODEL", _env("ai_groupmate__remote_media_embedding_model", "")))
    parser.add_argument("--remote-media-embedding-dimensions", type=int, default=_cfg_int(config, "REMOTE_MEDIA_EMBEDDING_DIMENSIONS", _env("ai_groupmate__remote_media_embedding_dimensions", ""), 2560))
    parser.add_argument("--remote-clip-base-url", default=_cfg(config, "REMOTE_CLIP_BASE_URL", _env("ai_groupmate__remote_clip_base_url", "")))
    parser.add_argument("--remote-clip-api-key", default=_cfg(config, "REMOTE_CLIP_API_KEY", _env("ai_groupmate__remote_clip_api_key", "")))
    parser.add_argument("--skip-chat", action="store_true", help="skip chat rebuild")
    parser.add_argument("--skip-media", action="store_true", help="skip media rebuild")
    parser.add_argument("--drop-chat", action="store_true", help="drop chat collection before rebuild")
    parser.add_argument("--drop-media", action="store_true", help="drop media collection before rebuild")
    parser.add_argument("--include-unvectorized-media", action="store_true", help="rebuild all non-blocked media rows, not just vectorized ones")
    parser.add_argument("--chat-batch-size", type=int, default=_cfg_int(config, "CHAT_BATCH_SIZE", default=100))
    parser.add_argument("--media-batch-size", type=int, default=_cfg_int(config, "MEDIA_BATCH_SIZE", default=20))
    parser.add_argument("--max-time-gap-minutes", type=int, default=_cfg_int(config, "MAX_TIME_GAP_MINUTES", default=60))
    parser.add_argument("--max-token-count", type=int, default=_cfg_int(config, "MAX_TOKEN_COUNT", default=1000))
    parser.add_argument("--max-messages", type=int, default=_cfg_int(config, "MAX_MESSAGES", default=50))
    parser.add_argument("--resume-chat-session", default="", help="resume from this chat session id")
    parser.add_argument("--resume-chat-batch", type=int, default=1, help="1-based batch index within resume chat session")
    parser.add_argument("--qdrant-max-retries", type=int, default=_cfg_int(config, "QDRANT_MAX_RETRIES", default=6))
    parser.add_argument("--qdrant-retry-sleep-seconds", type=float, default=float(_cfg(config, "QDRANT_RETRY_SLEEP_SECONDS", default="2")))
    parser.add_argument("--http-max-retries", type=int, default=_cfg_int(config, "HTTP_MAX_RETRIES", default=5))
    parser.add_argument("--http-retry-sleep-seconds", type=float, default=float(_cfg(config, "HTTP_RETRY_SLEEP_SECONDS", default="2")))
    parser.add_argument("--dry-run", action="store_true", help="calculate and print counts only")

    args = parser.parse_args(argv)
    if not args.include_unvectorized_media:
        args.include_unvectorized_media = _cfg_bool(config, "INCLUDE_UNVECTORIZED_MEDIA", default=False)
    if not args.dry_run:
        args.dry_run = _cfg_bool(config, "DRY_RUN", default=False)
    return args


def _build_chat_context(rows: list[ChatRow]) -> tuple[str, list[int]]:
    parts: list[str] = []
    msg_ids: list[int] = []
    for row in rows:
        parts.append(f"[{row.created_at.strftime('%Y-%m-%d %H:%M:%S')}] {row.user_name}: {row.content}")
        msg_ids.append(row.msg_id)
    return "\n".join(parts), msg_ids


def _group_chat_rows(
    rows: list[ChatRow],
    max_time_gap: timedelta,
    max_token_count: int,
    max_messages: int,
) -> list[tuple[str, list[int]]]:
    groups: list[tuple[str, list[int]]] = []
    current_rows: list[ChatRow] = []
    current_tokens = 0
    last_time: datetime | None = None

    for row in rows:
        row_text = f"[{row.created_at.strftime('%Y-%m-%d %H:%M:%S')}] {row.user_name}: {row.content}"
        row_tokens = _estimate_token_count(row_text)

        should_split = False
        if current_rows and last_time and row.created_at - last_time > max_time_gap:
            should_split = True
        elif current_rows and current_tokens + row_tokens > max_token_count:
            should_split = True
        elif current_rows and len(current_rows) >= max_messages:
            should_split = True

        if should_split:
            groups.append(_build_chat_context(current_rows))
            current_rows = []
            current_tokens = 0

        current_rows.append(row)
        current_tokens += row_tokens
        last_time = row.created_at

    if current_rows:
        groups.append(_build_chat_context(current_rows))
    return groups


def _list_session_ids(engine) -> list[str]:
    sql = text(f"SELECT DISTINCT session_id FROM {CHAT_TABLE} ORDER BY session_id")
    with engine.connect() as conn:
        return [str(row[0]) for row in conn.execute(sql).all() if row[0] is not None]


def _load_chat_rows(engine, session_id: str) -> list[ChatRow]:
    sql = text(
        f"""
        SELECT msg_id, session_id, user_name, content, created_at
        FROM {CHAT_TABLE}
        WHERE session_id = :session_id
        ORDER BY created_at, msg_id
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(sql, {"session_id": session_id}).mappings().all()
    return [
        ChatRow(
            msg_id=int(row["msg_id"]),
            session_id=str(row["session_id"]),
            user_name=str(row["user_name"] or ""),
            content=str(row["content"] or ""),
            created_at=row["created_at"],
        )
        for row in rows
    ]


def _load_media_rows(engine, include_unvectorized_media: bool) -> list[MediaRow]:
    conditions = ["blocked = false"]
    if not include_unvectorized_media:
        conditions.append("vectorized = true")

    sql = text(
        f"""
        SELECT media_id, file_path, description, blocked, vectorized, "references" AS media_references, created_at
        FROM {MEDIA_TABLE}
        WHERE {' AND '.join(conditions)}
        ORDER BY media_id
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(sql).mappings().all()
    return [
        MediaRow(
            media_id=int(row["media_id"]),
            file_path=str(row["file_path"] or ""),
            description=str(row["description"] or ""),
            blocked=bool(row["blocked"]),
            vectorized=bool(row["vectorized"]),
            references=int(row["media_references"] or 0),
            created_at=row["created_at"],
        )
        for row in rows
    ]


def _mark_chat_vectorized(engine, msg_ids: list[int]) -> None:
    if not msg_ids:
        return
    sql = text(f"UPDATE {CHAT_TABLE} SET vectorized = true WHERE msg_id IN :msg_ids").bindparams(bindparam("msg_ids", expanding=True))
    with engine.begin() as conn:
        conn.execute(sql, {"msg_ids": msg_ids})


def _mark_media_vectorized(engine, media_ids: list[int]) -> None:
    if not media_ids:
        return
    sql = text(f"UPDATE {MEDIA_TABLE} SET vectorized = true WHERE media_id IN :media_ids").bindparams(bindparam("media_ids", expanding=True))
    with engine.begin() as conn:
        conn.execute(sql, {"media_ids": media_ids})


def _resolve_media_path(pic_dir: Path, stored_path: str) -> Path:
    path = Path(stored_path)
    if path.is_absolute():
        return path
    return pic_dir / path


def _ensure_collection(client: QdrantClient, collection_name: str, vector_dim: int, create_session_index: bool = False) -> None:
    if client.collection_exists(collection_name):
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=vector_dim, distance=models.Distance.COSINE),
    )
    if create_session_index:
        client.create_payload_index(
            collection_name=collection_name,
            field_name="session_id",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )


def _stable_chat_point_id(session_id: str, context_text: str, msg_ids: list[int]) -> str:
    base = f"{session_id}\n{msg_ids[0] if msg_ids else 0}\n{msg_ids[-1] if msg_ids else 0}\n{context_text}"
    digest = hashlib.sha1(base.encode("utf-8")).hexdigest()
    return str(uuid.uuid5(uuid.NAMESPACE_URL, digest))


def _upsert_with_retry(
    client: QdrantClient,
    collection_name: str,
    points: list[models.PointStruct],
    max_retries: int,
    retry_sleep_seconds: float,
) -> None:
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            client.upsert(collection_name=collection_name, points=points, wait=True)
            return
        except Exception as e:
            last_error = e
            print(
                f"[qdrant] upsert failed collection={collection_name} attempt={attempt}/{max_retries} "
                f"error={type(e).__name__}: {e}"
            )
            if attempt >= max_retries:
                break
            time.sleep(retry_sleep_seconds * attempt)
    assert last_error is not None
    raise last_error


def _rebuild_chat(args: argparse.Namespace, engine, client: QdrantClient, model_client: RemoteModelClient) -> tuple[int, int]:
    session_ids = _list_session_ids(engine)
    total_contexts = 0
    total_messages = 0
    resume_active = bool(args.resume_chat_session)

    for session_id in session_ids:
        rows = _load_chat_rows(engine, session_id)
        if not rows:
            continue

        groups = _group_chat_rows(
            rows,
            max_time_gap=timedelta(minutes=args.max_time_gap_minutes),
            max_token_count=args.max_token_count,
            max_messages=args.max_messages,
        )
        total_contexts += len(groups)
        total_messages += len(rows)
        print(f"[chat] session={session_id} rows={len(rows)} contexts={len(groups)}")

        if args.dry_run or not groups:
            continue

        start_offset = 0
        if resume_active:
            if session_id != args.resume_chat_session:
                print(f"[chat] skipped session={session_id} before resume target")
                continue
            start_offset = max(0, (args.resume_chat_batch - 1) * args.chat_batch_size)
            print(
                f"[chat] resuming session={session_id} from batch={args.resume_chat_batch} "
                f"offset={start_offset}"
            )
            resume_active = False

        for start in range(start_offset, len(groups), args.chat_batch_size):
            batch = groups[start : start + args.chat_batch_size]
            texts = [item[0] for item in batch]
            vectors = model_client.embed_documents(texts)
            if len(vectors) != len(texts):
                raise RuntimeError(f"chat embedding count mismatch: {len(vectors)} != {len(texts)}")

            points = []
            msg_ids: list[int] = []
            now_ts = int(time.time())
            for (context_text, batch_msg_ids), vector in zip(batch, vectors):
                points.append(
                    models.PointStruct(
                        id=_stable_chat_point_id(session_id, context_text, batch_msg_ids),
                        vector=vector,
                        payload={
                            "session_id": session_id,
                            "text": context_text,
                            "created_at": now_ts,
                        },
                    )
                )
                msg_ids.extend(batch_msg_ids)

            _upsert_with_retry(
                client,
                args.chat_collection,
                points,
                max_retries=args.qdrant_max_retries,
                retry_sleep_seconds=args.qdrant_retry_sleep_seconds,
            )
            _mark_chat_vectorized(engine, msg_ids)
            print(f"[chat] upserted session={session_id} batch={start // args.chat_batch_size + 1} points={len(points)}")

    return total_messages, total_contexts


def _rebuild_media(args: argparse.Namespace, engine, client: QdrantClient, model_client: RemoteModelClient) -> tuple[int, int]:
    if not args.pic_dir:
        raise ValueError("--pic-dir is required for media rebuild")

    pic_dir = Path(args.pic_dir).expanduser()
    rows = _load_media_rows(engine, args.include_unvectorized_media)
    existing_rows: list[MediaRow] = []
    missing_files = 0

    for row in rows:
        file_path = _resolve_media_path(pic_dir, row.file_path)
        if file_path.exists():
            existing_rows.append(
                MediaRow(
                    media_id=row.media_id,
                    file_path=str(file_path),
                    description=row.description,
                    blocked=row.blocked,
                    vectorized=row.vectorized,
                    references=row.references,
                    created_at=row.created_at,
                )
            )
        else:
            missing_files += 1

    print(f"[media] selected={len(rows)} existing_files={len(existing_rows)} missing_files={missing_files}")
    if args.dry_run or not existing_rows:
        return len(existing_rows), missing_files

    processed_ids: list[int] = []
    for start in range(0, len(existing_rows), args.media_batch_size):
        batch = existing_rows[start : start + args.media_batch_size]
        image_paths = [row.file_path for row in batch]
        vectors = model_client.embed_media_images(image_paths)
        if len(vectors) != len(batch):
            raise RuntimeError(f"media embedding count mismatch: {len(vectors)} != {len(batch)}")

        points = []
        for row, vector in zip(batch, vectors):
            points.append(
                models.PointStruct(
                    id=row.media_id,
                    vector=vector,
                    payload={
                        "created_at": int(row.created_at.timestamp() * 1000),
                        "description": row.description,
                        "file_path": row.file_path,
                        "blocked": row.blocked,
                    },
                )
            )
            processed_ids.append(row.media_id)

        _upsert_with_retry(
            client,
            args.media_collection,
            points,
            max_retries=args.qdrant_max_retries,
            retry_sleep_seconds=args.qdrant_retry_sleep_seconds,
        )
        print(f"[media] upserted batch={start // args.media_batch_size + 1} points={len(points)}")

    _mark_media_vectorized(engine, processed_ids)
    return len(existing_rows), missing_files


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if not args.db_url:
        raise ValueError("db_url is required")
    if not args.qdrant_uri:
        raise ValueError("qdrant_uri is required")
    if args.skip_chat and args.skip_media:
        raise ValueError("cannot skip both chat and media rebuild")

    engine = create_engine(_normalize_db_url(args.db_url))
    client = QdrantClient(url=args.qdrant_uri, api_key=args.qdrant_api_key, timeout=60)
    model_client = RemoteModelClient(
        base_url=args.remote_model_base_url,
        api_key=args.remote_model_api_key,
        embedding_base_url=args.remote_embedding_base_url,
        embedding_api_key=args.remote_embedding_api_key,
        embedding_model=args.remote_embedding_model,
        embedding_dimensions=args.chat_vector_dim,
        media_embedding_provider=args.remote_media_embedding_provider,
        media_embedding_base_url=args.remote_media_embedding_base_url or args.remote_clip_base_url or args.remote_model_base_url,
        media_embedding_api_key=args.remote_media_embedding_api_key or args.remote_clip_api_key or args.remote_model_api_key,
        media_embedding_model=args.remote_media_embedding_model,
        media_embedding_dimensions=args.remote_media_embedding_dimensions or args.media_vector_dim,
        clip_base_url=args.remote_clip_base_url,
        clip_api_key=args.remote_clip_api_key,
        http_max_retries=args.http_max_retries,
        http_retry_sleep_seconds=args.http_retry_sleep_seconds,
    )

    if args.drop_chat and not args.dry_run and client.collection_exists(args.chat_collection):
        client.delete_collection(args.chat_collection)
        print(f"[chat] dropped collection {args.chat_collection}")
    if args.drop_media and not args.dry_run and client.collection_exists(args.media_collection):
        client.delete_collection(args.media_collection)
        print(f"[media] dropped collection {args.media_collection}")

    if not args.skip_chat and not args.dry_run:
        _ensure_collection(client, args.chat_collection, args.chat_vector_dim, create_session_index=True)
    if not args.skip_media and not args.dry_run:
        _ensure_collection(client, args.media_collection, args.media_vector_dim)

    chat_messages = 0
    chat_contexts = 0
    media_count = 0
    missing_files = 0

    if not args.skip_chat:
        chat_messages, chat_contexts = _rebuild_chat(args, engine, client, model_client)
    if not args.skip_media:
        media_count, missing_files = _rebuild_media(args, engine, client, model_client)

    print("==== rebuild summary ====")
    if not args.skip_chat:
        print(f"chat_messages={chat_messages}")
        print(f"chat_contexts={chat_contexts}")
    if not args.skip_media:
        print(f"media_vectors={media_count}")
        print(f"media_missing_files={missing_files}")
    print(f"dry_run={args.dry_run}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("aborted", file=sys.stderr)
        raise SystemExit(130)
