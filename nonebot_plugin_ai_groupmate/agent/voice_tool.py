import asyncio
import re
import time
from dataclasses import dataclass

import httpx
from langchain.tools import tool
from nonebot.log import logger
from nonebot_plugin_alconna import UniMessage
from nonebot_plugin_orm import get_session
from pydantic import BaseModel, Field

from ..config import ScopedConfig
from ..model import ChatHistory
from ..reply_guard import is_request_active

_health_lock = asyncio.Lock()
_tts_lock = asyncio.Lock()
_health_checked_at = 0.0
_health_result = False
_health_detail = "not checked"
_health_cache_key: tuple[str, str] | None = None
_health_text_langs: tuple[str, ...] = ()

_VOICE_MAX_TEXT_LENGTH = 120
_VOICE_TTS_PATH = "/tts"
_VOICE_HEALTH_PATH = "/"
_VOICE_HEALTH_TIMEOUT_SECONDS = 2.0
_VOICE_TTS_TIMEOUT_SECONDS = 180.0
_VOICE_HEALTH_CACHE_SECONDS = 30.0
_VOICE_UNHEALTHY_CACHE_SECONDS = 10.0


class VoiceToolError(Exception):
    """Voice tool failed in a controlled way."""


@dataclass(frozen=True)
class VoiceServiceHealthSnapshot:
    healthy: bool
    detail: str
    text_langs: tuple[str, ...]


class SendVoiceArgs(BaseModel):
    content: str = Field(description="要合成为语音的短文本。")
    text_lang: str | None = Field(
        default=None,
        description="目标文本语言。若健康检查返回可用列表，只能从返回的值里选。",
    )
    speed_factor: float | None = Field(default=None, description="语速。")
    top_k: int | None = Field(default=None, description="GPT-SoVITS top_k。")
    top_p: float | None = Field(default=None, description="GPT-SoVITS top_p。")
    temperature: float | None = Field(default=None, description="GPT-SoVITS temperature。")


def _build_url(base_url: str, path: str) -> str:
    path = (path or "").strip()
    if path.startswith(("http://", "https://")):
        return path

    base = (base_url or "").strip().rstrip("/")
    if not path:
        return base
    return f"{base}/{path.lstrip('/')}"


def _normalize_audio_format(value: str) -> str:
    return (value or "wav").strip().lower().lstrip(".")


def _is_voice_configured(config: ScopedConfig) -> bool:
    if not config.voice_enabled:
        return False
    if not (config.voice_base_url or "").strip():
        return False
    return True


def _parse_health_text_langs(response: httpx.Response) -> tuple[str, ...]:
    try:
        payload = response.json()
    except Exception:
        return ()

    if not isinstance(payload, dict):
        return ()

    raw_text_langs = payload.get("text_lang")
    if not isinstance(raw_text_langs, list):
        return ()

    normalized: list[str] = []
    seen: set[str] = set()
    for item in raw_text_langs:
        if not isinstance(item, str):
            continue
        value = item.strip()
        if not value or value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    return tuple(normalized)


def _current_health_cache_key(config: ScopedConfig) -> tuple[str, str]:
    return ((config.voice_base_url or "").strip(), _VOICE_HEALTH_PATH)


def get_voice_health_snapshot(config: ScopedConfig | None = None) -> VoiceServiceHealthSnapshot:
    if config is not None and _health_cache_key != _current_health_cache_key(config):
        return VoiceServiceHealthSnapshot(False, "not checked", ())
    return VoiceServiceHealthSnapshot(_health_result, _health_detail, _health_text_langs)


def get_voice_supported_text_langs(config: ScopedConfig | None = None) -> tuple[str, ...]:
    return get_voice_health_snapshot(config).text_langs


def _resolve_text_lang(config: ScopedConfig, text_lang: str | None) -> str:
    resolved = (text_lang or config.voice_text_lang or "").strip()
    if not resolved:
        raise VoiceToolError("text_lang 为空，无法发送语音。")

    available_text_langs = get_voice_supported_text_langs(config)
    if available_text_langs and resolved not in available_text_langs:
        available = ", ".join(available_text_langs)
        raise VoiceToolError(f"text_lang 不可用: {resolved}。当前可选值: {available}")
    return resolved


def _resolve_tts_payload(
    config: ScopedConfig,
    text: str,
    *,
    text_lang: str | None = None,
    speed_factor: float | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    temperature: float | None = None,
) -> dict[str, object]:
    return {
        "text": text,
        "text_lang": _resolve_text_lang(config, text_lang),
        "speed_factor": float(config.voice_speed_factor if speed_factor is None else speed_factor),
        "top_k": int(config.voice_top_k if top_k is None else top_k),
        "top_p": float(config.voice_top_p if top_p is None else top_p),
        "temperature": float(config.voice_temperature if temperature is None else temperature),
    }


async def is_voice_service_healthy(config: ScopedConfig, *, force: bool = False) -> bool:
    """Check GPT-SoVITS health with short timeout and TTL cache."""
    global _health_cache_key, _health_checked_at, _health_detail, _health_result, _health_text_langs

    if not _is_voice_configured(config):
        return False

    cache_key = _current_health_cache_key(config)
    now = time.monotonic()
    ttl = _VOICE_HEALTH_CACHE_SECONDS if _health_result else _VOICE_UNHEALTHY_CACHE_SECONDS
    ttl = max(float(ttl or 0), 0.0)

    if not force and _health_cache_key == cache_key and now - _health_checked_at < ttl:
        return _health_result

    async with _health_lock:
        now = time.monotonic()
        ttl = _VOICE_HEALTH_CACHE_SECONDS if _health_result else _VOICE_UNHEALTHY_CACHE_SECONDS
        ttl = max(float(ttl or 0), 0.0)
        if not force and _health_cache_key == cache_key and now - _health_checked_at < ttl:
            return _health_result

        url = _build_url(config.voice_base_url, _VOICE_HEALTH_PATH)
        had_cached_result = _health_cache_key == cache_key and _health_checked_at > 0
        old_result = _health_result
        try:
            timeout = httpx.Timeout(_VOICE_HEALTH_TIMEOUT_SECONDS)
            async with httpx.AsyncClient(
                timeout=timeout,
                follow_redirects=True,
                trust_env=False,
            ) as client:
                response = await client.get(url)
            healthy = response.status_code == 200
            detail = f"HTTP {response.status_code}"
            text_langs = _parse_health_text_langs(response) if healthy else ()
        except Exception as e:
            healthy = False
            detail = f"{type(e).__name__}: {e}"
            text_langs = ()

        _health_cache_key = cache_key
        _health_checked_at = time.monotonic()
        _health_result = healthy
        _health_detail = detail
        _health_text_langs = text_langs

        if not had_cached_result or healthy != old_result:
            if healthy:
                logger.info(f"语音服务健康检查通过: {url} ({detail})")
            else:
                logger.warning(f"语音服务健康检查失败: {url} ({detail})")
        return healthy


def get_voice_health_detail() -> str:
    return _health_detail


async def _request_tts_audio(
    config: ScopedConfig,
    text: str,
    *,
    text_lang: str | None = None,
    speed_factor: float | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    temperature: float | None = None,
) -> bytes:
    url = _build_url(config.voice_base_url, _VOICE_TTS_PATH)
    payload = _resolve_tts_payload(
        config,
        text,
        text_lang=text_lang,
        speed_factor=speed_factor,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
    )
    timeout = httpx.Timeout(_VOICE_TTS_TIMEOUT_SECONDS)

    async with _tts_lock:
        async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
            response = await client.post(url, json=payload)

    if response.status_code != 200:
        error_text = response.text.replace("\n", " ").strip()
        raise VoiceToolError(f"TTS 接口返回 HTTP {response.status_code}: {error_text[:300]}")

    if not response.content:
        raise VoiceToolError("TTS 接口返回了空音频")

    return bytes(response.content)


def _normalize_voice_text(text: str, max_length: int) -> str:
    normalized = re.sub(r"\s+", " ", text or "").strip()
    if max_length > 0 and len(normalized) > max_length:
        raise VoiceToolError(f"语音文本过长，请控制在 {max_length} 字以内")
    return normalized


async def _is_current_request_active(session_id: str, request_id: str | None) -> bool:
    if request_id is None:
        return True
    return await is_request_active(session_id, request_id)


def create_voice_tool(
    session_id: str,
    request_id: str | None,
    config: ScopedConfig,
    bot_name: str,
):
    """Create a GPT-SoVITS voice sending tool bound to the current session."""

    @tool("send_voice", args_schema=SendVoiceArgs)
    async def send_voice(
        content: str,
        text_lang: str | None = None,
        speed_factor: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        temperature: float | None = None,
    ) -> str:
        """
        将文本合成为语音并发送到当前群聊。
        只有用户明确要求你“发语音、用语音说、念出来、读出来”时才使用。

        Args:
            content: 要合成为语音的短文本。
            text_lang: 目标文本语言。
            speed_factor: 语速。
            top_k: GPT-SoVITS top_k。
            top_p: GPT-SoVITS top_p。
            temperature: GPT-SoVITS temperature。
        """
        if not await _is_current_request_active(session_id, request_id):
            return "请求已过期，已取消发送语音。"

        try:
            voice_text = _normalize_voice_text(content, _VOICE_MAX_TEXT_LENGTH)
            if not voice_text:
                return "语音文本为空，未发送。"

            if not await is_voice_service_healthy(config):
                return f"语音服务暂时不可用，未发送语音。健康状态: {get_voice_health_detail()}"

            audio = await _request_tts_audio(
                config,
                voice_text,
                text_lang=text_lang,
                speed_factor=speed_factor,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
            )

            if not await _is_current_request_active(session_id, request_id):
                return "请求已过期，已取消发送语音。"

            result = await UniMessage.voice(raw=audio, mimetype="audio/wav", name="voice.wav").send()
            msg_id = result.msg_ids[-1]["message_id"] if result.msg_ids else "unknown"
            async with get_session() as db_session:
                chat_history = ChatHistory(
                    session_id=session_id,
                    user_id=bot_name,
                    content_type="bot",
                    content=f"id: {msg_id}\n发送了一条语音，文本是: {voice_text}",
                    user_name=bot_name,
                )
                db_session.add(chat_history)
                await db_session.commit()

            logger.info(f"Bot已发送语音: {voice_text}")
            return f"语音已成功发送。实际语音文本：{voice_text}"
        except VoiceToolError as e:
            logger.warning(f"语音工具执行失败: {e}")
            return f"发送语音失败: {e}"
        except Exception as e:
            logger.error(f"语音工具异常: {e}")
            return f"发送语音失败: {type(e).__name__}: {e}"

    return send_voice
