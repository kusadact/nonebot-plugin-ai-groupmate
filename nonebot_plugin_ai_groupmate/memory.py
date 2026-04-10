import os
import io
import re
import time
import json
import uuid
import base64
import asyncio
import urllib.error
import urllib.request
import mimetypes
from typing import Any

from nonebot import get_plugin_config
from nonebot.log import logger
from qdrant_client import AsyncQdrantClient, models

from .config import Config

CHAT_COLLECTION = "chat_collection"
MEDIA_COLLECTION = "media_collection"
MEDIA_SEARCH_RECALL_LIMIT = 20
MEDIA_SEARCH_RETURN_LIMIT = 5
DASHSCOPE_MULTIMODAL_BATCH_LIMIT = 5

plugin_config = get_plugin_config(Config).ai_groupmate


def _image_to_base64(image_input: str) -> str:
    if image_input.startswith("data:image/"):
        _, _, payload = image_input.partition(",")
        if payload:
            return payload
    with open(image_input, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


class RemoteModelClient:
    def __init__(
        self,
        embedding_base_url: str = "",
        embedding_api_key: str = "",
        embedding_model: str = "",
        embedding_dimensions: int = 0,
        rerank_base_url: str = "",
        rerank_api_key: str = "",
        rerank_model: str = "",
        media_embedding_base_url: str = "",
        media_embedding_api_key: str = "",
        media_embedding_provider: str = "openai",
        media_embedding_model: str = "",
        media_embedding_dimensions: int = 0,
        media_rerank_provider: str = "openai",
        media_rerank_base_url: str = "",
        media_rerank_api_key: str = "",
        media_rerank_model: str = "",
    ):
        self.embedding_base_url = embedding_base_url.rstrip("/")
        self.embedding_api_key = embedding_api_key
        self.embedding_model = embedding_model
        self.embedding_dimensions = embedding_dimensions

        self.rerank_base_url = rerank_base_url.rstrip("/")
        self.rerank_api_key = rerank_api_key
        self.rerank_model = rerank_model

        self.media_embedding_base_url = media_embedding_base_url.rstrip("/")
        self.media_embedding_api_key = media_embedding_api_key
        self.media_embedding_provider = (media_embedding_provider or "openai").strip().lower()
        self.media_embedding_model = media_embedding_model
        self.media_embedding_dimensions = media_embedding_dimensions

        self.media_rerank_provider = (media_rerank_provider or "openai").strip().lower()
        self.media_rerank_base_url = media_rerank_base_url.rstrip("/")
        self.media_rerank_api_key = media_rerank_api_key
        self.media_rerank_model = media_rerank_model

    def _post_json_with_base(self, base_url: str, path: str, payload: dict[str, Any], api_key: str = "") -> dict[str, Any]:
        if not base_url:
            raise RuntimeError("remote base_url is empty")
        url = f"{base_url}{path}"
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8")
            except Exception:
                pass
            raise RuntimeError(f"remote http error {e.code}: {body or e.reason}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"remote request failed: {e}") from e
        return json.loads(body)

    @staticmethod
    def _normalize_embeddings_response(data: dict[str, Any]) -> dict[str, list[Any]]:
        dense: list[Any] = []
        for item in data.get("data", []):
            if isinstance(item, dict) and "embedding" in item:
                dense.append(item["embedding"])
        if not dense:
            raise RuntimeError(f"invalid embeddings response: {data}")
        return {"dense": dense}

    @staticmethod
    def _normalize_dashscope_embeddings_response(data: dict[str, Any]) -> dict[str, list[Any]]:
        dense: list[Any] = []
        output = data.get("output", {})
        for item in output.get("embeddings", []):
            if isinstance(item, dict) and "embedding" in item:
                dense.append(item["embedding"])
        if not dense:
            raise RuntimeError(f"invalid dashscope embeddings response: {data}")
        return {"dense": dense}

    @staticmethod
    def _normalize_rerank_results(data: dict[str, Any], fallback_documents: list[Any]) -> dict[str, list[dict[str, Any]]]:
        normalized = []
        for item in data.get("results", []):
            if not isinstance(item, dict):
                continue
            index = item.get("index")
            document = item.get("document")
            if document is None and isinstance(index, int) and 0 <= index < len(fallback_documents):
                document = fallback_documents[index]
            normalized.append(
                {
                    "index": index,
                    "score": float(item.get("relevance_score", item.get("score", 0))),
                    "document": document,
                }
            )
        return {"results": normalized}

    @staticmethod
    def _normalize_dashscope_rerank_results(data: dict[str, Any], fallback_documents: list[Any]) -> dict[str, list[dict[str, Any]]]:
        normalized = []
        output = data.get("output", {})
        for item in output.get("results", []):
            if not isinstance(item, dict):
                continue
            index = item.get("index")
            document = item.get("document")
            if document is None and isinstance(index, int) and 0 <= index < len(fallback_documents):
                document = fallback_documents[index]
            normalized.append(
                {
                    "index": index,
                    "score": float(item.get("relevance_score", item.get("score", 0))),
                    "document": document,
                }
            )
        return {"results": normalized}

    def _post_embeddings(self, base_url: str, api_key: str, model: str, inputs: list[Any], dimensions: int = 0) -> dict[str, list[Any]]:
        if not model:
            raise RuntimeError("remote embedding model is empty")
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
        raise RuntimeError("unsupported image source for dashscope")

    def _post_dashscope_media_embeddings(self, items: list[dict[str, str]]) -> dict[str, list[Any]]:
        if not self.media_embedding_model:
            raise RuntimeError("remote media embedding model is empty")
        dense: list[Any] = []
        for start in range(0, len(items), DASHSCOPE_MULTIMODAL_BATCH_LIMIT):
            payload: dict[str, Any] = {
                "model": self.media_embedding_model,
                "input": {"contents": items[start : start + DASHSCOPE_MULTIMODAL_BATCH_LIMIT]},
            }
            if self.media_embedding_dimensions > 0:
                payload["parameters"] = {"dimension": self.media_embedding_dimensions}
            data = self._post_json_with_base(
                self.media_embedding_base_url,
                "",
                payload,
                self.media_embedding_api_key,
            )
            dense.extend(self._normalize_dashscope_embeddings_response(data)["dense"])
        return {"dense": dense}

    def embed_documents(self, texts: list[str]) -> dict[str, list[Any]]:
        if not self.embedding_base_url:
            raise RuntimeError("remote_embedding_base_url is empty")
        return self._post_embeddings(
            self.embedding_base_url,
            self.embedding_api_key,
            self.embedding_model,
            texts,
            self.embedding_dimensions,
        )

    def rerank(self, query: str, texts: list[str]) -> dict[str, list[dict[str, Any]]]:
        if not self.rerank_base_url:
            raise RuntimeError("remote_rerank_base_url is empty")
        if not self.rerank_model:
            raise RuntimeError("remote_rerank_model is empty")
        payload = {
            "model": self.rerank_model,
            "query": query,
            "documents": texts,
            "top_n": len(texts),
            "return_documents": True,
        }
        data = self._post_json_with_base(
            self.rerank_base_url,
            "/rerank",
            payload,
            self.rerank_api_key,
        )
        normalized = []
        for item in self._normalize_rerank_results(data, texts).get("results", []):
            doc = item.get("document")
            text = doc if isinstance(doc, str) else ""
            if isinstance(doc, dict):
                text = doc.get("text", "")
            normalized.append({"text": text, "score": item["score"], "index": item.get("index")})
        return {"results": normalized}

    @staticmethod
    def _build_image_input(image_base64: str) -> list[dict[str, str]]:
        return [{"type": "input_image", "image_url": f"data:image/png;base64,{image_base64}"}]

    def embed_media_texts(self, texts: list[str]) -> dict[str, list[Any]]:
        if not self.media_embedding_base_url:
            raise RuntimeError("remote_media_embedding_base_url is empty")
        if self.media_embedding_provider == "aliyun_dashscope":
            items = [{"text": text} for text in texts]
            return self._post_dashscope_media_embeddings(items)
        return self._post_embeddings(
            self.media_embedding_base_url,
            self.media_embedding_api_key,
            self.media_embedding_model,
            texts,
            self.media_embedding_dimensions,
        )

    def embed_media_images_base64(self, images_base64: list[str]) -> dict[str, list[Any]]:
        if not self.media_embedding_base_url:
            raise RuntimeError("remote_media_embedding_base_url is empty")
        if self.media_embedding_provider == "aliyun_dashscope":
            items = [{"image": self._to_dashscope_image(f"data:image/png;base64,{item}")} for item in images_base64]
            return self._post_dashscope_media_embeddings(items)
        inputs = [self._build_image_input(item) for item in images_base64]
        return self._post_embeddings(
            self.media_embedding_base_url,
            self.media_embedding_api_key,
            self.media_embedding_model,
            inputs,
            self.media_embedding_dimensions,
        )

    def has_media_rerank(self) -> bool:
        return bool(self.media_rerank_base_url)

    def rerank_media(self, query: Any, documents: list[Any]) -> dict[str, list[dict[str, Any]]]:
        if not self.media_rerank_base_url:
            raise RuntimeError("remote_media_rerank_base_url is empty")
        if not self.media_rerank_model:
            raise RuntimeError("remote_media_rerank_model is empty")

        if self.media_rerank_provider == "aliyun_dashscope":
            dashscope_query: dict[str, str] | None = None
            if isinstance(query, str) and query.strip():
                dashscope_query = {"text": query.strip()}
            elif isinstance(query, list):
                for part in query:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") == "input_image":
                        dashscope_query = {"image": str(part.get("image_url", ""))}
                        break
                    if part.get("type") == "input_text":
                        dashscope_query = {"text": str(part.get("text", "")).strip()}
                        break
            if not dashscope_query:
                raise RuntimeError("dashscope media rerank expects text or image query")

            dashscope_docs: list[Any] = []
            for doc in documents:
                if isinstance(doc, dict):
                    if "image" in doc or "text" in doc or "video" in doc:
                        dashscope_docs.append(doc)
                        continue
                if isinstance(doc, str):
                    text_value = doc.strip()
                    if text_value:
                        dashscope_docs.append({"text": text_value})
                    continue
                if isinstance(doc, list):
                    chosen: dict[str, str] | None = None
                    for part in doc:
                        if not isinstance(part, dict):
                            continue
                        if part.get("type") == "input_image":
                            chosen = {"image": str(part.get("image_url", ""))}
                            break
                    if chosen is None:
                        for part in doc:
                            if not isinstance(part, dict):
                                continue
                            if part.get("type") == "input_text":
                                chosen = {"text": str(part.get("text", "")).strip()}
                                break
                    if chosen:
                        dashscope_docs.append(chosen)

            payload: dict[str, Any] = {
                "model": self.media_rerank_model,
                "input": {"query": dashscope_query, "documents": dashscope_docs},
                "parameters": {"return_documents": True, "top_n": len(dashscope_docs)},
            }
            data = self._post_json_with_base(
                self.media_rerank_base_url,
                "",
                payload,
                self.media_rerank_api_key,
            )
            return self._normalize_dashscope_rerank_results(data, dashscope_docs)

        payload = {
            "model": self.media_rerank_model,
            "query": query,
            "documents": documents,
            "top_n": len(documents),
            "return_documents": True,
        }
        data = self._post_json_with_base(
            self.media_rerank_base_url,
            "/rerank",
            payload,
            self.media_rerank_api_key,
        )
        return self._normalize_rerank_results(data, documents)


class VectorDBOperator:
    def __init__(self):
        self.enabled = bool((plugin_config.qdrant_uri or "").strip())
        self.chat_col = CHAT_COLLECTION
        self.media_col = MEDIA_COLLECTION
        self.chat_vector_dim = plugin_config.remote_embedding_dimensions or plugin_config.chat_vector_dim
        self.media_vector_dim = plugin_config.remote_media_embedding_dimensions or plugin_config.media_vector_dim
        self.client: AsyncQdrantClient | None = None
        self._collections_ready = False
        self._init_lock = asyncio.Lock()

        if not self.enabled:
            logger.info("未配置 qdrant_uri，Qdrant 向量库功能已禁用")
            self.remote_client = None
            return

        self.client = AsyncQdrantClient(
            url=plugin_config.qdrant_uri,
            api_key=plugin_config.qdrant_api_key,
            timeout=60,
        )
        self.remote_client = RemoteModelClient(
            embedding_base_url=plugin_config.remote_embedding_base_url,
            embedding_api_key=plugin_config.remote_embedding_api_key,
            embedding_model=plugin_config.remote_embedding_model,
            embedding_dimensions=self.chat_vector_dim,
            rerank_base_url=plugin_config.remote_rerank_base_url,
            rerank_api_key=plugin_config.remote_rerank_api_key,
            rerank_model=plugin_config.remote_rerank_model,
            media_embedding_provider=plugin_config.remote_media_embedding_provider,
            media_embedding_base_url=plugin_config.remote_media_embedding_base_url,
            media_embedding_api_key=plugin_config.remote_media_embedding_api_key,
            media_embedding_model=plugin_config.remote_media_embedding_model,
            media_embedding_dimensions=self.media_vector_dim,
            media_rerank_provider=plugin_config.remote_media_rerank_provider,
            media_rerank_base_url=plugin_config.remote_media_rerank_base_url,
            media_rerank_api_key=plugin_config.remote_media_rerank_api_key,
            media_rerank_model=plugin_config.remote_media_rerank_model,
        )

    @staticmethod
    def _ensure_vector_dim(vector: Any, expected_dim: int, label: str) -> Any:
        actual_dim = len(vector)
        if actual_dim != expected_dim:
            raise RuntimeError(f"{label} dim mismatch: expected {expected_dim}, got {actual_dim}")
        return vector

    async def _ensure_collections(self) -> None:
        if not self.enabled or self.client is None:
            return
        if self._collections_ready:
            return

        async with self._init_lock:
            if self._collections_ready:
                return

            if not await self.client.collection_exists(self.chat_col):
                await self.client.create_collection(
                    collection_name=self.chat_col,
                    vectors_config=models.VectorParams(size=self.chat_vector_dim, distance=models.Distance.COSINE),
                )
                await self.client.create_payload_index(
                    collection_name=self.chat_col,
                    field_name="session_id",
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )

            if not await self.client.collection_exists(self.media_col):
                await self.client.create_collection(
                    collection_name=self.media_col,
                    vectors_config=models.VectorParams(size=self.media_vector_dim, distance=models.Distance.COSINE),
                )

            self._collections_ready = True

    async def healthcheck(self) -> tuple[bool, str]:
        if not self.enabled or self.client is None:
            return False, "disabled"
        try:
            await asyncio.wait_for(self.client.get_collections(), timeout=3.0)
            return True, "ok"
        except asyncio.TimeoutError:
            return False, "timeout"
        except Exception as e:
            return False, type(e).__name__

    async def _embed_documents(self, texts: list[str], batch_size: int = 50) -> list[list[float]]:
        if not self.remote_client:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            encoded = await asyncio.to_thread(self.remote_client.embed_documents, chunk)
            dense_vectors = [self._ensure_vector_dim(vector, self.chat_vector_dim, "chat embedding") for vector in encoded["dense"]]
            vectors.extend(dense_vectors)
        return vectors

    async def batch_insert(self, texts: list[str], session_id: str) -> None:
        if not self.enabled or self.client is None or not texts:
            return
        await self._ensure_collections()

        vectors = await self._embed_documents(texts)
        if len(vectors) != len(texts):
            raise RuntimeError(f"embedding count mismatch: {len(vectors)} != {len(texts)}")

        current_time = int(time.time())
        points = []
        for text, vector in zip(texts, vectors):
            points.append(
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "session_id": session_id,
                        "text": text,
                        "created_at": current_time,
                    },
                )
            )

        await self.client.upsert(collection_name=self.chat_col, points=points, wait=True)

    async def insert(self, text, session_id, collection_name=CHAT_COLLECTION):
        await self.batch_insert([text], session_id)

    async def search(self, text, search_filter: str = "", collection_name=CHAT_COLLECTION, with_meta: bool = False):
        if not self.enabled or self.client is None or not text:
            return {"texts": [], "vector_ids": []} if with_meta else []
        await self._ensure_collections()

        encoded = await asyncio.to_thread(self.remote_client.embed_documents, text)
        dense_vector = self._ensure_vector_dim(encoded["dense"][0], self.chat_vector_dim, "chat query embedding")

        session_filter = None
        if search_filter:
            match = re.search(r'session_id\s*==\s*[\'"](.+?)[\'"]', search_filter)
            if match:
                session_filter = models.Filter(
                    must=[models.FieldCondition(key="session_id", match=models.MatchValue(value=match.group(1)))]
                )

        search_result = await self.client.query_points(
            collection_name=collection_name,
            query=dense_vector,
            query_filter=session_filter,
            limit=20,
            with_payload=True,
        )

        candidates: list[dict[str, Any]] = []
        for point in getattr(search_result, "points", []) or []:
            payload = point.payload or {}
            text_value = payload.get("text")
            if not isinstance(text_value, str) or not text_value:
                continue
            candidates.append({"id": str(point.id), "text": text_value})

        if not candidates:
            return {"texts": [], "vector_ids": []} if with_meta else []

        ordered = candidates
        if self.remote_client and len(candidates) > 1:
            rerank = await asyncio.to_thread(self.remote_client.rerank, text[0], [item["text"] for item in candidates])
            ranked: list[dict[str, Any]] = []
            seen: set[int] = set()
            for item in rerank.get("results", []):
                index = item.get("index")
                if not isinstance(index, int) or not (0 <= index < len(candidates)) or index in seen:
                    continue
                ranked.append(candidates[index])
                seen.add(index)
            for index, candidate in enumerate(candidates):
                if index not in seen:
                    ranked.append(candidate)
            ordered = ranked

        texts = [item["text"] for item in ordered]
        vector_ids = [item["id"] for item in ordered]
        return {"texts": texts, "vector_ids": vector_ids} if with_meta else texts

    async def search_chat(self, query: str, session_id: str) -> str:
        result = await self.search([query], search_filter=f'session_id == "{session_id}"', with_meta=True)
        return "\n".join(result.get("texts", [])) if isinstance(result, dict) else "\n".join(result)

    async def _get_media_records(self, ids: list[int]) -> list[dict[str, Any]]:
        if not self.client or not ids:
            return []
        records = await self.client.retrieve(
            collection_name=self.media_col,
            ids=ids,
            with_payload=True,
            with_vectors=False,
        )
        ordered: list[dict[str, Any]] = []
        by_id = {int(record.id): record for record in records}
        for media_id in ids:
            record = by_id.get(int(media_id))
            payload = getattr(record, "payload", {}) or {}
            ordered.append(
                {
                    "id": int(media_id),
                    "description": str(payload.get("description", "") or ""),
                    "file_path": str(payload.get("file_path", "") or ""),
                }
            )
        return ordered

    @staticmethod
    def _build_media_rerank_documents(records: list[dict[str, Any]]) -> list[Any]:
        documents: list[Any] = []
        for record in records:
            parts: list[dict[str, str]] = []
            description = str(record.get("description", "") or "").strip()
            if description:
                parts.append({"type": "input_text", "text": description})
            file_path = str(record.get("file_path", "") or "").strip()
            if file_path and os.path.exists(file_path):
                mime_type = mimetypes.guess_type(file_path)[0] or "image/png"
                with open(file_path, "rb") as image_file:
                    payload = base64.b64encode(image_file.read()).decode("utf-8")
                parts.append({"type": "input_image", "image_url": f"data:{mime_type};base64,{payload}"})
            documents.append(parts if parts else description)
        return documents

    async def _rerank_media_candidates(self, query: Any, ids: list[int]) -> list[int]:
        ordered_ids = [int(item) for item in ids]
        if not ordered_ids:
            return []
        if not self.remote_client or not self.remote_client.has_media_rerank() or len(ordered_ids) <= 1:
            return ordered_ids[:MEDIA_SEARCH_RETURN_LIMIT]
        if self.remote_client.media_rerank_provider == "aliyun_dashscope" and not isinstance(query, str):
            return ordered_ids[:MEDIA_SEARCH_RETURN_LIMIT]

        records = await self._get_media_records(ordered_ids)
        documents = await asyncio.to_thread(self._build_media_rerank_documents, records)
        rerank_res = await asyncio.to_thread(self.remote_client.rerank_media, query, documents)
        reranked_ids: list[int] = []
        seen: set[int] = set()
        for item in rerank_res.get("results", []):
            index = item.get("index")
            if not isinstance(index, int) or not (0 <= index < len(records)):
                continue
            media_id = int(records[index]["id"])
            if media_id in seen:
                continue
            reranked_ids.append(media_id)
            seen.add(media_id)
        for media_id in ordered_ids:
            if media_id not in seen:
                reranked_ids.append(media_id)
        return reranked_ids[:MEDIA_SEARCH_RETURN_LIMIT]

    async def insert_media(self, media_id, image_urls, description: str = "", file_path: str = "", blocked: bool = False, collection_name=MEDIA_COLLECTION):
        if not self.enabled or not self.client or not image_urls:
            return
        await self._ensure_collections()

        images_base64 = await asyncio.to_thread(lambda: [_image_to_base64(item) for item in image_urls])
        image_embeddings = await asyncio.to_thread(self.remote_client.embed_media_images_base64, images_base64)
        dense_vector = self._ensure_vector_dim(image_embeddings["dense"][0], self.media_vector_dim, "media embedding")

        await self.client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=int(media_id),
                    vector=dense_vector,
                    payload={
                        "created_at": int(time.time() * 1000),
                        "description": description,
                        "file_path": file_path,
                        "blocked": bool(blocked),
                    },
                )
            ],
            wait=True,
        )

    async def delete_media(self, media_id: int) -> None:
        if not self.enabled or not self.client:
            return
        await self._ensure_collections()
        await self.client.delete(
            collection_name=self.media_col,
            points_selector=models.PointIdsList(points=[int(media_id)]),
            wait=True,
        )

    async def search_media(self, text):
        if not self.enabled or not self.client or not text:
            return []
        await self._ensure_collections()

        text_embeddings = await asyncio.to_thread(self.remote_client.embed_media_texts, text)
        dense_vector = self._ensure_vector_dim(text_embeddings["dense"][0], self.media_vector_dim, "media text query embedding")
        res = await self.client.query_points(
            collection_name=self.media_col,
            query=dense_vector,
            query_filter=models.Filter(
                must_not=[models.FieldCondition(key="blocked", match=models.MatchValue(value=True))]
            ),
            limit=MEDIA_SEARCH_RECALL_LIMIT,
            with_payload=False,
        )
        candidate_ids = [int(item.id) for item in getattr(res, "points", []) or []]
        return await self._rerank_media_candidates(text[0], candidate_ids)

    async def search_media_by_pic(self, image_urls):
        if not self.enabled or not self.client or not image_urls:
            return []
        await self._ensure_collections()

        images_base64 = await asyncio.to_thread(lambda: [_image_to_base64(item) for item in image_urls])
        image_embeddings = await asyncio.to_thread(self.remote_client.embed_media_images_base64, images_base64)
        dense_vector = self._ensure_vector_dim(image_embeddings["dense"][0], self.media_vector_dim, "media image query embedding")
        res = await self.client.query_points(
            collection_name=self.media_col,
            query=dense_vector,
            query_filter=models.Filter(
                must_not=[models.FieldCondition(key="blocked", match=models.MatchValue(value=True))]
            ),
            limit=MEDIA_SEARCH_RECALL_LIMIT,
            with_payload=False,
        )
        candidate_ids = [int(item.id) for item in getattr(res, "points", []) or []]
        if not self.remote_client or not self.remote_client.has_media_rerank():
            return candidate_ids[:MEDIA_SEARCH_RETURN_LIMIT]
        query = self.remote_client._build_image_input(images_base64[0])
        return await self._rerank_media_candidates(query, candidate_ids)


DB = VectorDBOperator()
