import base64
import mimetypes
import os
import time
import hashlib
import asyncio

from openai import AsyncOpenAI
from nonebot import logger, get_plugin_config

from .config import Config

plugin_config = get_plugin_config(Config).ai_groupmate

ollama_client = None
openai_client = None
OllamaImage = None
OllamaMessage = None

if plugin_config.vlm_provider == "ollama":
    try:
        from ollama import Image as OllamaImage, Message as OllamaMessage, AsyncClient as OllamaClient
    except Exception as e:
        logger.error(f"Ollama not available: {e}")
        OllamaClient = None

    if OllamaClient:
        ollama_client = OllamaClient(host=plugin_config.vlm_ollama_base_url, timeout=120)
elif plugin_config.vlm_provider == "openai":
    openai_client = AsyncOpenAI(api_key=plugin_config.vlm_openai_api_key, base_url=plugin_config.vlm_openai_base_url, timeout=120.0)


def encode_image_to_base64(file_path: str) -> str:
    """将本地图片转换为 Base64 字符串"""
    with open(file_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_mime_type(file_path: str) -> str:
    """获取文件的 mime type，默认为 image/jpeg"""
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type if mime_type else "image/jpeg"


def _safe_hash_file(file_path: str) -> str:
    try:
        with open(file_path, "rb") as f:
            data = f.read(64 * 1024)
        return hashlib.sha256(data).hexdigest()[:12]
    except Exception:
        return "unknown"


async def image_vl(
    file_path,
    prompt: str = "请描述一下这个图片",
    max_tokens: int = 1024,
    timeout_s: float | None = None,
) -> str | None:
    start_ts = time.time()
    file_path_str = str(file_path)
    mime_type = get_mime_type(file_path_str)
    file_size = -1
    try:
        file_size = os.path.getsize(file_path_str)
    except Exception:
        pass
    file_hash = _safe_hash_file(file_path_str)
    logger.info(
        f"[vlm] start file={file_path_str} size={file_size} mime={mime_type} hash={file_hash} provider={plugin_config.vlm_provider} model={plugin_config.vlm_model} max_tokens={max_tokens} timeout={timeout_s}"
    )
    try:
        if plugin_config.vlm_provider == "ollama":
            if not ollama_client:
                logger.error("Ollama client not initialized")
                return None

            if not OllamaImage or not OllamaMessage:
                logger.error("Ollama classes not available")
                return None

            coro = ollama_client.chat(
                model=plugin_config.vlm_model,
                messages=[OllamaMessage(role="user", content=prompt, images=[OllamaImage(value=file_path)])],
                options={"repeat_penalty": 1.5, "num_ctx": 1024},
            )
            response = await (asyncio.wait_for(coro, timeout=timeout_s) if timeout_s else coro)
            content = response.message.content

        elif plugin_config.vlm_provider == "openai":
            if not openai_client:
                logger.error("OpenAI client not initialized")
                return None

            # 1. 获取 Base64 和 MimeType
            base64_image = encode_image_to_base64(file_path_str)

            # 2. 构造 OpenAI 多模态消息格式
            coro = openai_client.chat.completions.create(
                model=plugin_config.vlm_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
                            },
                        ],
                    }
                ],
                max_tokens=max_tokens,
            )
            response = await (asyncio.wait_for(coro, timeout=timeout_s) if timeout_s else coro)
            content = response.choices[0].message.content

        else:
            logger.error(f"Unknown provider: {plugin_config.vlm_provider}")
            return None

        # 统一的后处理逻辑
        elapsed = round(time.time() - start_ts, 2)
        if not content:
            logger.warning(f"[vlm] empty response file={file_path_str} elapsed={elapsed}s")
            return None

        # 防止输出重复的内容或过长
        if len(content) > 2000:
            logger.warning(f"[vlm] response too long file={file_path_str} chars={len(content)} elapsed={elapsed}s")
            return None

        logger.info(f"[vlm] done file={file_path_str} chars={len(content)} elapsed={elapsed}s")

        return content

    except asyncio.TimeoutError:
        elapsed = round(time.time() - start_ts, 2)
        logger.error(f"[vlm] timeout file={file_path_str} elapsed={elapsed}s timeout={timeout_s}")
        return None
    except Exception as e:
        elapsed = round(time.time() - start_ts, 2)
        logger.error(f"[vlm] error file={file_path_str} elapsed={elapsed}s error={e}")
        return None
