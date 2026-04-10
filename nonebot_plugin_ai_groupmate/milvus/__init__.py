import os
import time
import json
import asyncio
import urllib.request
import urllib.error
import base64
import io
import mimetypes
from typing import Any, TYPE_CHECKING

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from nonebot import logger, get_driver, get_plugin_config
from pymilvus import DataType, Function, FunctionType, MilvusClient, WeightedRanker, AnnSearchRequest, AsyncMilvusClient
if TYPE_CHECKING:
    from pymilvus.model.hybrid import BGEM3EmbeddingFunction
    from pymilvus.model.reranker import BGERerankFunction

from ..config import Config

LOCAL_TEXT_VECTOR_DIM = 1024
LOCAL_MEDIA_VECTOR_DIM = 1024
MEDIA_SEARCH_RECALL_LIMIT = 20
MEDIA_SEARCH_RETURN_LIMIT = 5
DEFAULT_MILVUS_DB_NAME = "ai_groupmate"
CHAT_COLLECTION = "chat_collection"
MEDIA_COLLECTION = "media_collection"
DASHSCOPE_MULTIMODAL_BATCH_LIMIT = 5


class RemoteModelClient:
    def __init__(
        self,
        base_url: str = "",
        api_key: str = "",
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
        clip_base_url: str = "",
        clip_api_key: str = "",
    ):
        # 兼容旧版：统一入口（/embed /rerank /clip）
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

        # 新版分路：text embedding / text rerank / media embedding / media rerank
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

        self.clip_base_url = clip_base_url.rstrip("/")
        self.clip_api_key = clip_api_key

    def _post_json_with_base(self, base_url: str, path: str, payload: dict, api_key: str = "") -> dict:
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

    def _post_json(self, path: str, payload: dict) -> dict:
        if not self.base_url:
            raise RuntimeError("remote_model_base_url is empty")
        return self._post_json_with_base(self.base_url, path, payload, self.api_key)

    @staticmethod
    def _normalize_embeddings_response(data: dict) -> dict:
        dense: list[Any] = []
        for item in data.get("data", []):
            if isinstance(item, dict) and "embedding" in item:
                dense.append(item["embedding"])
        if not dense:
            raise RuntimeError(f"invalid embeddings response: {data}")
        return {"dense": dense}

    @staticmethod
    def _normalize_dashscope_embeddings_response(data: dict) -> dict:
        dense: list[Any] = []
        output = data.get("output", {})
        for item in output.get("embeddings", []):
            if isinstance(item, dict) and "embedding" in item:
                dense.append(item["embedding"])
        if not dense:
            raise RuntimeError(f"invalid dashscope embeddings response: {data}")
        return {"dense": dense}

    @staticmethod
    def _normalize_rerank_results(data: dict, fallback_documents: list[Any]) -> dict:
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
    def _normalize_dashscope_rerank_results(data: dict, fallback_documents: list[Any]) -> dict:
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

    def _post_embeddings(self, base_url: str, api_key: str, model: str, inputs: list[Any], dimensions: int = 0) -> dict:
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
            with open(image_input, "rb") as f:
                payload = base64.b64encode(f.read()).decode("utf-8")
            return f"data:{mime_type};base64,{payload}"
        raise RuntimeError("unsupported image source for dashscope")

    def _post_dashscope_media_embeddings(self, items: list[dict[str, str]]) -> dict:
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
                self.media_embedding_api_key or self.api_key,
            )
            dense.extend(self._normalize_dashscope_embeddings_response(data)["dense"])
        return {"dense": dense}

    def _post_clip_json(self, path: str, payload: dict) -> dict:
        # 旧版 clip 接口兼容层，未配置 OpenAI 风格媒体接口时回退使用。
        clip_base = self.clip_base_url or self.base_url
        clip_key = self.clip_api_key or self.api_key
        if not clip_base:
            raise RuntimeError("remote_clip_base_url is empty")
        return self._post_json_with_base(clip_base, path, payload, clip_key)

    # 统一返回：{"dense": [[...], ...]}
    def embed_documents(self, texts: list[str]) -> dict:
        # 配置了 embedding 分路时，按硅基流动/OpenAI 风格协议调用。
        if self.embedding_base_url:
            return self._post_embeddings(
                self.embedding_base_url,
                self.embedding_api_key or self.api_key,
                self.embedding_model,
                texts,
                self.embedding_dimensions,
            )
        return self._post_json("/embed", {"texts": texts})

    # 统一返回：{"results": [{"text": "...", "score": 0.1}, ...]}
    def rerank(self, query: str, texts: list[str]) -> dict:
        # 配置了 rerank 分路时，按硅基流动协议调用。
        if self.rerank_base_url:
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
                self.rerank_api_key or self.api_key,
            )
            normalized = []
            for item in self._normalize_rerank_results(data, texts).get("results", []):
                doc = item.get("document")
                text = doc if isinstance(doc, str) else ""
                if isinstance(doc, dict):
                    text = doc.get("text", "")
                normalized.append({"text": text, "score": item["score"]})
            return {"results": normalized}
        return self._post_json("/rerank", {"query": query, "texts": texts})

    @staticmethod
    def _build_image_input(image_base64: str) -> list[dict[str, str]]:
        return [{"type": "input_image", "image_url": f"data:image/png;base64,{image_base64}"}]

    def embed_media_texts(self, texts: list[str]) -> dict:
        if self.media_embedding_base_url:
            if self.media_embedding_provider == "aliyun_dashscope":
                items = [{"text": text} for text in texts]
                return self._post_dashscope_media_embeddings(items)
            return self._post_embeddings(
                self.media_embedding_base_url,
                self.media_embedding_api_key or self.api_key,
                self.media_embedding_model,
                texts,
                self.media_embedding_dimensions,
            )
        return self._post_clip_json("/clip/text", {"texts": texts})

    def embed_media_images_base64(self, images_base64: list[str]) -> dict:
        if self.media_embedding_base_url:
            if self.media_embedding_provider == "aliyun_dashscope":
                items = [{"image": self._to_dashscope_image(f"data:image/png;base64,{item}")} for item in images_base64]
                return self._post_dashscope_media_embeddings(items)
            inputs = [self._build_image_input(item) for item in images_base64]
            return self._post_embeddings(
                self.media_embedding_base_url,
                self.media_embedding_api_key or self.api_key,
                self.media_embedding_model,
                inputs,
                self.media_embedding_dimensions,
            )
        return self._post_clip_json("/clip/image", {"images_base64": images_base64})

    def has_media_rerank(self) -> bool:
        return bool(self.media_rerank_base_url)

    def rerank_media(self, query: Any, documents: list[Any]) -> dict:
        if not self.media_rerank_base_url:
            raise RuntimeError("remote_media_rerank_base_url is empty")
        if not self.media_rerank_model:
            raise RuntimeError("remote_media_rerank_model is empty")
        if self.media_rerank_provider == "aliyun_dashscope":
            dashscope_query: dict[str, str] | None = None
            if isinstance(query, str) and query.strip():
                dashscope_query = {"text": query.strip()}
            elif isinstance(query, list):
                # Expect list of parts from _build_image_input
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
                self.media_rerank_api_key or self.api_key,
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
            self.media_rerank_api_key or self.api_key,
        )
        return self._normalize_rerank_results(data, documents)


def _image_to_base64(image_input) -> str:
    from PIL import Image as PILImage

    img = None
    close_after = False
    if isinstance(image_input, str):
        img = PILImage.open(image_input)
        close_after = True
    else:
        img = image_input

    try:
        if getattr(img, "is_animated", False):
            img.seek(0)
    except Exception:
        pass

    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    if close_after:
        try:
            img.close()
        except Exception:
            pass
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _image_to_data_url(image_input) -> str:
    return f"data:image/png;base64,{_image_to_base64(image_input)}"


class MilvusOperator:
    client: MilvusClient | None
    async_client: AsyncMilvusClient | None
    ef: Any | None
    bge_rf: Any | None
    ranker: WeightedRanker | None

    def __init__(self, uri: str = "http://localhost:19530", user: str = "", password: str = "", db_name: str = DEFAULT_MILVUS_DB_NAME):
        # 1. __init__ 中只保存配置，不连接数据库，不加载模型
        self.uri = uri
        self.user = user
        self.password = password
        self.db_name = db_name.strip()
        self.semaphore = asyncio.Semaphore(1)

        # 将模型和客户端占位符设为 None
        self.ef = None
        self.bge_rf = None
        self.clip_model = None
        self.remote_client = None
        self.client = None
        self.async_client = None
        self.ranker = None
        self.chat_vector_dim = LOCAL_TEXT_VECTOR_DIM
        self.media_vector_dim = LOCAL_MEDIA_VECTOR_DIM
        self.initialized = False

    def _should_use_named_database(self) -> bool:
        normalized = (self.uri or "").strip().lower()
        return "://" in normalized and not normalized.endswith(".db")

    @staticmethod
    def _call_database_method(client: Any, method_names: tuple[str, ...], *args, **kwargs) -> Any:
        last_error: Exception | None = None
        for method_name in method_names:
            method = getattr(client, method_name, None)
            if method is None:
                continue
            try:
                return method(*args, **kwargs)
            except TypeError as e:
                last_error = e
                if kwargs and args:
                    continue
                if kwargs:
                    try:
                        return method(*args or tuple(kwargs.values()))
                    except TypeError as inner_e:
                        last_error = inner_e
                        continue
        if last_error:
            raise last_error
        raise AttributeError(f"database method not found: {method_names}")

    def _ensure_database_selected(self, client: Any) -> None:
        if not self.db_name or not self._should_use_named_database():
            return
        try:
            database_names = self._call_database_method(client, ("list_databases",))
        except Exception as e:
            logger.warning(f"Milvus 数据库命名空间不可用，回退使用默认库: {e}")
            return

        if self.db_name not in set(database_names or []):
            self._call_database_method(client, ("create_database",), db_name=self.db_name)
            logger.info(f"Milvus database created: {self.db_name}")

        self._call_database_method(client, ("use_database", "using_database"), db_name=self.db_name)

    def _build_sync_client(self) -> MilvusClient:
        return MilvusClient(self.uri, self.user, self.password)

    @staticmethod
    def _ensure_vector_dim(vector: Any, expected_dim: int, label: str) -> Any:
        actual_dim = len(vector)
        if actual_dim != expected_dim:
            raise RuntimeError(f"{label} dim mismatch: expected {expected_dim}, got {actual_dim}")
        return vector

    @staticmethod
    def _ordered_media_records(ids: list[int], records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        by_id: dict[int, dict[str, Any]] = {}
        for item in records:
            try:
                item_id = int(item.get("id"))
            except Exception:
                continue
            by_id[item_id] = item
        ordered = []
        for item_id in ids:
            ordered.append(
                {
                    "id": int(item_id),
                    "description": str((by_id.get(int(item_id)) or {}).get("description", "") or ""),
                    "file_path": str((by_id.get(int(item_id)) or {}).get("file_path", "") or ""),
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
                parts.append({"type": "input_image", "image_url": _image_to_data_url(file_path)})
            documents.append(parts if parts else description)
        return documents

    async def init_models(self):
        """
        2. 创建一个专门的初始化方法，在 NoneBot 启动时调用
        """
        if self.initialized:
            return

        logger.info("Loading Milvus models and database connection...")
        try:
            text_embedding_dimensions = plugin_config.remote_embedding_dimensions or plugin_config.chat_vector_dim
            media_embedding_dimensions = plugin_config.remote_media_embedding_dimensions or plugin_config.media_vector_dim
            use_remote = any(
                [
                    bool(plugin_config.remote_model_base_url),
                    bool(plugin_config.remote_embedding_base_url),
                    bool(plugin_config.remote_rerank_base_url),
                    bool(plugin_config.remote_media_embedding_base_url),
                    bool(plugin_config.remote_media_rerank_base_url),
                    bool(plugin_config.remote_clip_base_url),
                ]
            )
            if use_remote:
                self.chat_vector_dim = plugin_config.chat_vector_dim
                self.media_vector_dim = plugin_config.media_vector_dim
                self.remote_client = RemoteModelClient(
                    base_url=plugin_config.remote_model_base_url,
                    api_key=plugin_config.remote_model_api_key,
                    embedding_base_url=plugin_config.remote_embedding_base_url,
                    embedding_api_key=plugin_config.remote_embedding_api_key,
                    embedding_model=plugin_config.remote_embedding_model,
                    embedding_dimensions=text_embedding_dimensions,
                    rerank_base_url=plugin_config.remote_rerank_base_url,
                    rerank_api_key=plugin_config.remote_rerank_api_key,
                    rerank_model=plugin_config.remote_rerank_model,
                    media_embedding_provider=plugin_config.remote_media_embedding_provider,
                    media_embedding_base_url=plugin_config.remote_media_embedding_base_url or plugin_config.remote_clip_base_url or plugin_config.remote_model_base_url,
                    media_embedding_api_key=plugin_config.remote_media_embedding_api_key or plugin_config.remote_clip_api_key,
                    media_embedding_model=plugin_config.remote_media_embedding_model,
                    media_embedding_dimensions=media_embedding_dimensions,
                    media_rerank_provider=plugin_config.remote_media_rerank_provider,
                    media_rerank_base_url=plugin_config.remote_media_rerank_base_url or plugin_config.remote_media_embedding_base_url or plugin_config.remote_clip_base_url or plugin_config.remote_model_base_url,
                    media_rerank_api_key=plugin_config.remote_media_rerank_api_key,
                    media_rerank_model=plugin_config.remote_media_rerank_model,
                    clip_base_url=plugin_config.remote_clip_base_url,
                    clip_api_key=plugin_config.remote_clip_api_key,
                )
                self.ef = None
                self.bge_rf = None
                self.clip_model = None
                logger.info("Using remote model service.")
            else:
                # Local fallback (kept for backward compatibility)
                import torch
                from transformers import AutoModel
                from pymilvus.model.hybrid import BGEM3EmbeddingFunction
                from pymilvus.model.reranker import BGERerankFunction

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                self.ef = BGEM3EmbeddingFunction(
                    model_name="BAAI/bge-m3",
                    device="cuda" if torch.cuda.is_available() else "cpu",
                )
                self.bge_rf = BGERerankFunction(
                    model_name="BAAI/bge-reranker-v2-m3",
                    device="cuda" if torch.cuda.is_available() else "cpu",
                )
                self.clip_model = AutoModel.from_pretrained("jinaai/jina-clip-v2", trust_remote_code=True).to(device)
                if self.clip_model:
                    self.clip_model.eval()
                self.chat_vector_dim = LOCAL_TEXT_VECTOR_DIM
                self.media_vector_dim = LOCAL_MEDIA_VECTOR_DIM
            self.client = self._build_sync_client()
            self._ensure_database_selected(self.client)
            self.ranker = WeightedRanker(0.8, 0.3)

            # 初始化 Collections (逻辑保持不变)
            self._init_collections()

            self.initialized = True
            logger.success("Milvus models ready and database connected.")

        except Exception as e:
            logger.error(f"Milvus 初始化失败: {e}")
            raise e

    def _init_collections(self):
        """将创建 Collection 的逻辑抽离出来"""
        assert self.client is not None, "MilvusClient is not initialized"

        if not self.client.has_collection(collection_name=CHAT_COLLECTION):
            schema = MilvusClient.create_schema(
                auto_id=True,
                enable_dynamic_field=True,
            )
            schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
            schema.add_field(field_name="session_id", datatype=DataType.VARCHAR, max_length=1000)
            schema.add_field(
                field_name="text",
                datatype=DataType.VARCHAR,
                enable_analyzer=True,
                max_length=20000,
            )
            # Define a sparse vector field to generate spare vectors with BM25
            schema.add_field(field_name="sparse", datatype=DataType.SPARSE_FLOAT_VECTOR)
            schema.add_field(field_name="dense", datatype=DataType.FLOAT_VECTOR, dim=self.chat_vector_dim)
            schema.add_field(field_name="created_at", datatype=DataType.INT64)
            bm25_function = Function(
                name="text_bm25_emb",  # Function name
                input_field_names=["text"],  # Name of the VARCHAR field containing raw text data
                output_field_names=["sparse"],
                # Name of the SPARSE_FLOAT_VECTOR field reserved to store generated embeddings
                function_type=FunctionType.BM25,
            )

            schema.add_function(bm25_function)
            # Prepare index parameters
            index_params = self.client.prepare_index_params()

            # Add indexes
            index_params.add_index(
                field_name="dense",
                index_name="dense_index",
                index_type="IVF_FLAT",
                metric_type="IP",
                params={"nlist": 128},
            )

            index_params.add_index(
                field_name="sparse",
                index_name="sparse_index",
                index_type="SPARSE_INVERTED_INDEX",  # Index type for sparse vectors
                metric_type="BM25",  # Set to `BM25` when using function to generate sparse vectors
                params={"inverted_index_algo": "DAAT_MAXSCORE"},
            )
            self.client.create_collection(
                collection_name=CHAT_COLLECTION,
                schema=schema,
                index_params=index_params,
            )

        if not self.client.has_collection(collection_name=MEDIA_COLLECTION):
            schema = MilvusClient.create_schema(
                enable_dynamic_field=True,
            )
            # Add fields to schema
            schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
            schema.add_field(field_name="dense", datatype=DataType.FLOAT_VECTOR, dim=self.media_vector_dim)
            schema.add_field(field_name="created_at", datatype=DataType.INT64)
            # Prepare index parameters
            index_params = self.client.prepare_index_params()

            # Add indexes
            index_params.add_index(field_name="dense", index_name="dense_index", index_type="AUTOINDEX", metric_type="IP")
            self.client.create_collection(
                collection_name=MEDIA_COLLECTION,
                schema=schema,
                index_params=index_params,
            )

    def _get_async_client(self) -> AsyncMilvusClient:
        """确保在当前运行的 loop 中获取或创建 client"""
        if self.async_client is None:
            if self.client is None:
                self.client = self._build_sync_client()
            self._ensure_database_selected(self.client)
            if self.db_name and self._should_use_named_database():
                try:
                    self.async_client = AsyncMilvusClient(self.uri, self.user, self.password, db_name=self.db_name)
                except TypeError as e:
                    raise RuntimeError("当前 pymilvus 版本不支持 AsyncMilvusClient(db_name=...)，请升级 pymilvus。") from e
            else:
                self.async_client = AsyncMilvusClient(self.uri, self.user, self.password)
        return self.async_client

    async def _get_media_records(self, ids: list[int]) -> list[dict[str, Any]]:
        if not ids:
            return []
        client = self._get_async_client()
        records = await client.get(collection_name=MEDIA_COLLECTION, ids=ids, output_fields=["description", "file_path"])
        return self._ordered_media_records(ids, records)

    async def _rerank_media_candidates(self, query: Any, ids: list[int]) -> list[int]:
        ordered_ids = [int(item) for item in ids]
        if not ordered_ids:
            return []
        if not self.remote_client or not self.remote_client.has_media_rerank() or len(ordered_ids) <= 1:
            return ordered_ids[:MEDIA_SEARCH_RETURN_LIMIT]
        if self.remote_client.media_rerank_provider == "aliyun_dashscope" and not isinstance(query, str):
            # DashScope qwen3-vl-rerank expects a text query; skip rerank for image query.
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

    async def insert(self, text, session_id, collection_name=CHAT_COLLECTION):
        # 安全检查：确保初始化完成
        if not self.initialized:
            logger.warning("MilvusOperator 尚未初始化，正在尝试初始化...")
            await self.init_models()

        client = self._get_async_client()
        if self.remote_client:
            encoded = await asyncio.to_thread(self.remote_client.embed_documents, [text])
            dense_vector = self._ensure_vector_dim(encoded["dense"][0], self.chat_vector_dim, "chat embedding")
        else:
            assert self.ef is not None, "Embedding function not initialized"
            async with self.semaphore:
                encoded = await asyncio.to_thread(self.ef.encode_documents, [text])
            dense_vector = self._ensure_vector_dim(encoded["dense"][0], self.chat_vector_dim, "chat embedding")
        data = {
            "session_id": session_id,
            "text": text,
            "dense": dense_vector,
            "created_at": int(time.time() * 1000),
        }
        res = await client.insert(collection_name=collection_name, data=data)
        return res

    async def batch_insert(self, texts, session_id, collection_name=CHAT_COLLECTION):
        if not self.initialized:
            await self.init_models()

        if not texts:
            return []
        if self.remote_client:
            encoded = await asyncio.to_thread(self.remote_client.embed_documents, texts)
            dense_vectors = [self._ensure_vector_dim(vector, self.chat_vector_dim, "chat embedding") for vector in encoded["dense"]]
        else:
            assert self.ef is not None, "Embedding function not initialized"
            async with self.semaphore:
                encoded = await asyncio.to_thread(self.ef.encode_documents, texts)
            dense_vectors = [self._ensure_vector_dim(vector, self.chat_vector_dim, "chat embedding") for vector in encoded["dense"]]

        batch_data = []
        current_time = int(time.time() * 1000)

        for i, text in enumerate(texts):
            data_item = {
                "session_id": session_id,
                "text": text,
                "dense": dense_vectors[i],
                "created_at": current_time,
            }
            batch_data.append(data_item)

        client = self._get_async_client()
        res = await client.insert(collection_name=collection_name, data=batch_data)
        return res

    async def insert_media(self, media_id, image_urls, description: str = "", file_path: str = "", collection_name=MEDIA_COLLECTION):
        if not self.initialized:
            await self.init_models()
        if self.remote_client:
            images_base64 = await asyncio.to_thread(lambda: [_image_to_base64(i) for i in image_urls])
            image_embeddings = await asyncio.to_thread(self.remote_client.embed_media_images_base64, images_base64)
            dense_vector = self._ensure_vector_dim(image_embeddings["dense"][0], self.media_vector_dim, "media embedding")
        else:
            assert self.clip_model is not None, "CLIP model not initialized"
            async with self.semaphore:
                image_embeddings = await asyncio.to_thread(self.clip_model.encode_image, image_urls)
            dense_vector = self._ensure_vector_dim(image_embeddings[0], self.media_vector_dim, "media embedding")
        data = {
            "id": media_id,
            "dense": dense_vector,
            "description": description,
            "file_path": file_path or (str(image_urls[0]) if image_urls else ""),
            "created_at": int(time.time() * 1000),
        }
        client = self._get_async_client()
        res = await client.insert(collection_name=collection_name, data=data)
        return res

    async def search(
        self,
        text: list[str],
        search_filter: str | None = None,
        collection_name=CHAT_COLLECTION,
        with_meta: bool = False,
    ):
        if not self.initialized:
            await self.init_models()
        assert self.ranker is not None, "Ranker not initialized"
        if self.remote_client:
            encoded = await asyncio.to_thread(self.remote_client.embed_documents, text)
            dense_vector = self._ensure_vector_dim(encoded["dense"][0], self.chat_vector_dim, "chat query embedding")
        else:
            assert self.ef is not None, "Embedding function not initialized"
            assert self.bge_rf is not None, "Reranker not initialized"
            async with self.semaphore:
                encoded = await asyncio.to_thread(self.ef.encode_documents, text)
            dense_vector = self._ensure_vector_dim(encoded["dense"][0], self.chat_vector_dim, "chat query embedding")

        search_param_1 = {
            "data": [dense_vector],
            "anns_field": "dense",
            "param": {"nprobe": 10},
            "limit": 10,
            "expr": search_filter,
        }
        request_1 = AnnSearchRequest(**search_param_1)

        search_param_2 = {
            "data": text,
            "anns_field": "sparse",
            "param": {"drop_ratio_search": 0.2},
            "limit": 10,
            "expr": search_filter,
        }
        request_2 = AnnSearchRequest(**search_param_2)
        reqs = [request_1, request_2]

        client = self._get_async_client()
        res = await client.hybrid_search(
            collection_name=collection_name,
            reqs=reqs,
            ranker=self.ranker,
            limit=10,
        )

        # 快速检查结果，避免后续空列表报错
        if not res or not res[0]:
            if with_meta:
                return {"texts": [], "vector_ids": []}
            return None

        ids = [i["id"] for i in res[0]]

        texts = await client.get(collection_name=collection_name, ids=ids, output_fields=["text"])

        text_list = [i["text"] for i in texts]
        if self.remote_client:
            rerank = await asyncio.to_thread(self.remote_client.rerank, text[0], text_list)
            results = rerank.get("results", [])
        else:
            async with self.semaphore:
                results = await asyncio.to_thread(self.bge_rf, text[0], text_list)

        if not results:
            if with_meta:
                return {"texts": [], "vector_ids": ids}
            return None

        if self.remote_client:
            best_texts = [i["text"] for i in results]
        else:
            best_texts = [i.text for i in results]
        if with_meta:
            return {"texts": best_texts, "vector_ids": ids}
        return best_texts

    async def search_media(self, text):
        if not self.initialized:
            await self.init_models()
        if self.remote_client:
            text_embeddings = await asyncio.to_thread(self.remote_client.embed_media_texts, text)
            dense_vector = self._ensure_vector_dim(text_embeddings["dense"][0], self.media_vector_dim, "media text query embedding")
        else:
            assert self.clip_model is not None, "CLIP model not initialized"
            async with self.semaphore:
                text_embeddings = await asyncio.to_thread(self.clip_model.encode_text, text)
            dense_vector = self._ensure_vector_dim(text_embeddings[0], self.media_vector_dim, "media text query embedding")
        client = self._get_async_client()
        res = await client.search(
            collection_name=MEDIA_COLLECTION,
            anns_field="dense",
            data=[dense_vector],
            limit=MEDIA_SEARCH_RECALL_LIMIT,
            search_params={"metric_type": "IP"},
        )
        candidate_ids = [int(i["id"]) for i in res[0]]
        return await self._rerank_media_candidates(text[0], candidate_ids)

    async def search_media_by_pic(self, image_urls):
        if not self.initialized:
            await self.init_models()
        if self.remote_client:
            images_base64 = await asyncio.to_thread(lambda: [_image_to_base64(i) for i in image_urls])
            image_embeddings = await asyncio.to_thread(self.remote_client.embed_media_images_base64, images_base64)
            dense_vector = self._ensure_vector_dim(image_embeddings["dense"][0], self.media_vector_dim, "media image query embedding")
        else:
            assert self.clip_model is not None, "CLIP model not initialized"
            async with self.semaphore:
                image_embeddings = await asyncio.to_thread(self.clip_model.encode_image, image_urls)
            dense_vector = self._ensure_vector_dim(image_embeddings[0], self.media_vector_dim, "media image query embedding")
        client = self._get_async_client()
        res = await client.search(
            collection_name=MEDIA_COLLECTION,
            anns_field="dense",
            data=[dense_vector],
            limit=MEDIA_SEARCH_RECALL_LIMIT,
            search_params={"metric_type": "IP"}
        )
        candidate_ids = [int(i["id"]) for i in res[0]]
        if not self.remote_client or not self.remote_client.has_media_rerank():
            return candidate_ids[:MEDIA_SEARCH_RETURN_LIMIT]
        query = RemoteModelClient._build_image_input(await asyncio.to_thread(_image_to_base64, image_urls[0]))
        return await self._rerank_media_candidates(query, candidate_ids)

plugin_config = get_plugin_config(Config).ai_groupmate

MilvusOP = MilvusOperator(
    plugin_config.milvus_uri,
    plugin_config.milvus_user,
    plugin_config.milvus_password,
    plugin_config.milvus_db_name,
)

# 4. 获取驱动器并注册启动钩子
driver = get_driver()


@driver.on_startup
async def _():
    # 5. 在机器人启动时才真正下载模型、连接数据库
    # Do not block NoneBot startup on Milvus connectivity.
    # Models/DB are initialized lazily on first use.
    logger.info("Milvus init deferred until first use.")
