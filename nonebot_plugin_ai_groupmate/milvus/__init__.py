import os
import time
import json
import asyncio
import urllib.request
import urllib.error
import base64
import io
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
        clip_base_url: str = "",
        clip_api_key: str = "",
    ):
        # 兼容旧版：统一入口（/embed /rerank /clip）
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

        # 新版分路：embedding / rerank / clip 分别配置
        self.embedding_base_url = embedding_base_url.rstrip("/")
        self.embedding_api_key = embedding_api_key
        self.embedding_model = embedding_model
        self.embedding_dimensions = embedding_dimensions

        self.rerank_base_url = rerank_base_url.rstrip("/")
        self.rerank_api_key = rerank_api_key
        self.rerank_model = rerank_model

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

    def _post_clip_json(self, path: str, payload: dict) -> dict:
        # clip 优先走分路地址，未配置时回退到旧统一入口。
        clip_base = self.clip_base_url or self.base_url
        clip_key = self.clip_api_key or self.api_key
        if not clip_base:
            raise RuntimeError("remote_clip_base_url is empty")
        return self._post_json_with_base(clip_base, path, payload, clip_key)

    # 统一返回：{"dense": [[...], ...]}
    def embed_documents(self, texts: list[str]) -> dict:
        # 配置了 embedding 分路时，按硅基流动/OpenAI 风格协议调用。
        if self.embedding_base_url:
            if not self.embedding_model:
                raise RuntimeError("remote_embedding_model is empty")
            payload: dict[str, Any] = {
                "model": self.embedding_model,
                "input": texts,
            }
            if self.embedding_dimensions > 0:
                payload["dimensions"] = self.embedding_dimensions
            data = self._post_json_with_base(
                self.embedding_base_url,
                "/v1/embeddings",
                payload,
                self.embedding_api_key or self.api_key,
            )
            dense: list[Any] = []
            for item in data.get("data", []):
                if isinstance(item, dict) and "embedding" in item:
                    dense.append(item["embedding"])
            if not dense:
                raise RuntimeError(f"invalid embeddings response: {data}")
            return {"dense": dense}
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
                "/v1/rerank",
                payload,
                self.rerank_api_key or self.api_key,
            )
            normalized = []
            for item in data.get("results", []):
                if not isinstance(item, dict):
                    continue
                score = item.get("relevance_score", item.get("score", 0))
                text = ""
                doc = item.get("document")
                if isinstance(doc, dict):
                    text = doc.get("text", "")
                elif isinstance(doc, str):
                    text = doc
                if not text:
                    idx = item.get("index")
                    if isinstance(idx, int) and 0 <= idx < len(texts):
                        text = texts[idx]
                normalized.append({"text": text, "score": float(score)})
            return {"results": normalized}
        return self._post_json("/rerank", {"query": query, "texts": texts})

    # 统一返回：{"dense": [[...], ...]}
    def clip_text(self, texts: list[str]) -> dict:
        return self._post_clip_json("/clip/text", {"texts": texts})

    # 统一返回：{"dense": [[...], ...]}
    def clip_image(self, image_urls: list[str]) -> dict:
        return self._post_clip_json("/clip/image", {"images": image_urls})

    # 统一返回：{"dense": [[...], ...]}
    def clip_image_base64(self, images_base64: list[str]) -> dict:
        return self._post_clip_json("/clip/image", {"images_base64": images_base64})


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


class MilvusOperator:
    client: MilvusClient | None
    async_client: AsyncMilvusClient | None
    ef: Any | None
    bge_rf: Any | None
    ranker: WeightedRanker | None

    def __init__(self, uri: str = "http://localhost:19530", user: str = "", password: str = ""):
        # 1. __init__ 中只保存配置，不连接数据库，不加载模型
        self.uri = uri
        self.user = user
        self.password = password
        self.semaphore = asyncio.Semaphore(1)

        # 将模型和客户端占位符设为 None
        self.ef = None
        self.bge_rf = None
        self.clip_model = None
        self.remote_client = None
        self.client = None
        self.async_client = None
        self.ranker = None
        self.initialized = False

    async def init_models(self):
        """
        2. 创建一个专门的初始化方法，在 NoneBot 启动时调用
        """
        if self.initialized:
            return

        logger.info("Loading Milvus models and database connection...")
        try:
            use_remote = any(
                [
                    bool(plugin_config.remote_model_base_url),
                    bool(plugin_config.remote_embedding_base_url),
                    bool(plugin_config.remote_rerank_base_url),
                    bool(plugin_config.remote_clip_base_url),
                ]
            )
            if use_remote:
                self.remote_client = RemoteModelClient(
                    base_url=plugin_config.remote_model_base_url,
                    api_key=plugin_config.remote_model_api_key,
                    embedding_base_url=plugin_config.remote_embedding_base_url,
                    embedding_api_key=plugin_config.remote_embedding_api_key,
                    embedding_model=plugin_config.remote_embedding_model,
                    embedding_dimensions=plugin_config.remote_embedding_dimensions,
                    rerank_base_url=plugin_config.remote_rerank_base_url,
                    rerank_api_key=plugin_config.remote_rerank_api_key,
                    rerank_model=plugin_config.remote_rerank_model,
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
            self.client = MilvusClient(self.uri, self.user, self.password)
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

        if not self.client.has_collection(collection_name="chat_collection"):
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
            schema.add_field(field_name="dense", datatype=DataType.FLOAT_VECTOR, dim=1024)
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
                collection_name="chat_collection",
                schema=schema,
                index_params=index_params,
            )

        if not self.client.has_collection(collection_name="media_collection"):
            schema = MilvusClient.create_schema(
                enable_dynamic_field=True,
            )
            # Add fields to schema
            schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
            schema.add_field(field_name="dense", datatype=DataType.FLOAT_VECTOR, dim=1024)
            schema.add_field(field_name="created_at", datatype=DataType.INT64)
            # Prepare index parameters
            index_params = self.client.prepare_index_params()

            # Add indexes
            index_params.add_index(field_name="dense", index_name="dense_index", index_type="AUTOINDEX", metric_type="IP")
            self.client.create_collection(
                collection_name="media_collection",
                schema=schema,
                index_params=index_params,
            )

    def _get_async_client(self) -> AsyncMilvusClient:
        """确保在当前运行的 loop 中获取或创建 client"""
        if self.async_client is None:
            self.async_client = AsyncMilvusClient(self.uri, self.user, self.password)
        return self.async_client

    async def insert(self, text, session_id, collection_name="chat_collection"):
        # 安全检查：确保初始化完成
        if not self.initialized:
            logger.warning("MilvusOperator 尚未初始化，正在尝试初始化...")
            await self.init_models()

        client = self._get_async_client()
        if self.remote_client:
            encoded = await asyncio.to_thread(self.remote_client.embed_documents, [text])
            dense_vector = encoded["dense"][0]
        else:
            assert self.ef is not None, "Embedding function not initialized"
            async with self.semaphore:
                encoded = await asyncio.to_thread(self.ef.encode_documents, [text])
            dense_vector = encoded["dense"][0]
        data = {
            "session_id": session_id,
            "text": text,
            "dense": dense_vector,
            "created_at": int(time.time() * 1000),
        }
        res = await client.insert(collection_name=collection_name, data=data)
        return res

    async def batch_insert(self, texts, session_id, collection_name="chat_collection"):
        if not self.initialized:
            await self.init_models()

        if not texts:
            return []
        if self.remote_client:
            encoded = await asyncio.to_thread(self.remote_client.embed_documents, texts)
            dense_vectors = encoded["dense"]
        else:
            assert self.ef is not None, "Embedding function not initialized"
            async with self.semaphore:
                encoded = await asyncio.to_thread(self.ef.encode_documents, texts)
            dense_vectors = encoded["dense"]

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

    async def insert_media(self, media_id, image_urls, collection_name="media_collection"):
        if not self.initialized:
            await self.init_models()
        if self.remote_client:
            images_base64 = await asyncio.to_thread(lambda: [_image_to_base64(i) for i in image_urls])
            image_embeddings = await asyncio.to_thread(self.remote_client.clip_image_base64, images_base64)
            dense_vector = image_embeddings["dense"][0]
        else:
            assert self.clip_model is not None, "CLIP model not initialized"
            async with self.semaphore:
                image_embeddings = await asyncio.to_thread(self.clip_model.encode_image, image_urls)
            dense_vector = image_embeddings[0]
        data = {
            "id": media_id,
            "dense": dense_vector,
            "created_at": int(time.time() * 1000),
        }
        client = self._get_async_client()
        res = await client.insert(collection_name=collection_name, data=data)
        return res

    async def search(
        self,
        text: list[str],
        search_filter: str | None = None,
        collection_name="chat_collection",
    ):
        if not self.initialized:
            await self.init_models()
        assert self.ranker is not None, "Ranker not initialized"
        if self.remote_client:
            encoded = await asyncio.to_thread(self.remote_client.embed_documents, text)
            dense_vector = encoded["dense"][0]
        else:
            assert self.ef is not None, "Embedding function not initialized"
            assert self.bge_rf is not None, "Reranker not initialized"
            async with self.semaphore:
                encoded = await asyncio.to_thread(self.ef.encode_documents, text)
            dense_vector = encoded["dense"][0]

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
            return None

        if self.remote_client:
            best_texts = [i["text"] for i in results]
        else:
            best_texts = [i.text for i in results]
        return best_texts

    async def search_media(self, text):
        if not self.initialized:
            await self.init_models()
        if self.remote_client:
            text_embeddings = await asyncio.to_thread(self.remote_client.clip_text, text)
            dense_vector = text_embeddings["dense"][0]
        else:
            assert self.clip_model is not None, "CLIP model not initialized"
            async with self.semaphore:
                text_embeddings = await asyncio.to_thread(self.clip_model.encode_text, text)
            dense_vector = text_embeddings[0]
        client = self._get_async_client()
        res = await client.search(collection_name="media_collection", anns_field="dense", data=[dense_vector], limit=3, search_params={"metric_type": "IP"})
        return [i["id"] for i in res[0]]

    async def search_media_by_pic(self, image_urls):
        if not self.initialized:
            await self.init_models()
        if self.remote_client:
            images_base64 = await asyncio.to_thread(lambda: [_image_to_base64(i) for i in image_urls])
            image_embeddings = await asyncio.to_thread(self.remote_client.clip_image_base64, images_base64)
            dense_vector = image_embeddings["dense"][0]
        else:
            assert self.clip_model is not None, "CLIP model not initialized"
            async with self.semaphore:
                image_embeddings = await asyncio.to_thread(self.clip_model.encode_image, image_urls)
            dense_vector = image_embeddings[0]
        client = self._get_async_client()
        res = await client.search(
            collection_name="media_collection",
            anns_field="dense",
            data=[dense_vector],
            limit=5,
            search_params={"metric_type": "IP"}
        )
        return [i["id"] for i in res[0]]

plugin_config = get_plugin_config(Config).ai_groupmate

MilvusOP = MilvusOperator(plugin_config.milvus_uri, plugin_config.milvus_user, plugin_config.milvus_password)

# 4. 获取驱动器并注册启动钩子
driver = get_driver()


@driver.on_startup
async def _():
    # 5. 在机器人启动时才真正下载模型、连接数据库
    # Do not block NoneBot startup on Milvus connectivity.
    # Models/DB are initialized lazily on first use.
    logger.info("Milvus init deferred until first use.")
