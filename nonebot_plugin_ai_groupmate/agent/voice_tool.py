import asyncio
import contextlib
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

import httpx
from langchain.tools import tool
from nonebot import get_bot
from nonebot.adapters.onebot.v11 import Message, MessageSegment
from nonebot.log import logger
from nonebot_plugin_alconna import UniMessage
from nonebot_plugin_orm import get_session

from ..config import ScopedConfig
from ..model import ChatHistory
from ..reply_guard import is_request_active

_health_lock = asyncio.Lock()
_tts_lock = asyncio.Lock()
_health_checked_at = 0.0
_health_result = False
_health_detail = "not checked"
_health_cache_key: tuple[str, str] | None = None
_ffmpeg_encoder_cache: dict[tuple[str, str], tuple[float, bool, str]] = {}

_MIME_BY_FORMAT = {
    "amr": "audio/amr",
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "ogg": "audio/ogg",
}


class VoiceToolError(Exception):
    """Voice tool failed in a controlled way."""


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


def _audio_mimetype(audio_format: str) -> str:
    return _MIME_BY_FORMAT.get(_normalize_audio_format(audio_format), f"audio/{_normalize_audio_format(audio_format)}")


def _normalize_voice_send_method(value: str) -> str:
    return (value or "raw").strip().lower()


def _build_ffmpeg_audio_filter(config: ScopedConfig) -> str | None:
    audio_filter = (config.voice_ffmpeg_audio_filter or "").strip()
    if audio_filter:
        return audio_filter

    volume_gain = max(float(config.voice_volume_gain or 0), 0.0)
    if volume_gain <= 0 or volume_gain == 1.0:
        return None
    return f"volume={volume_gain:g}"


def _is_voice_configured(config: ScopedConfig) -> bool:
    if not config.voice_enabled:
        return False
    if not (config.voice_base_url or "").strip():
        return False
    if not (config.voice_ref_audio_path or "").strip():
        return False
    if not (config.voice_prompt_text or "").strip():
        return False
    return True


def _send_requirement_detail(config: ScopedConfig) -> str | None:
    method = _normalize_voice_send_method(config.voice_send_method)
    if method == "raw":
        return None
    if method != "path":
        return f"不支持的语音发送方式: {config.voice_send_method}"

    send_dir = (config.voice_send_dir or "").strip()
    if not send_dir:
        return "voice_send_method=path 时必须配置 voice_send_dir"

    try:
        directory = Path(send_dir)
        directory.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        return f"无法创建语音发送目录 {send_dir}: {type(e).__name__}: {e}"

    if not directory.is_dir():
        return f"语音发送目录不是有效目录: {send_dir}"
    return None


def _conversion_requirement_detail(config: ScopedConfig) -> str | None:
    source_format = _normalize_audio_format(config.voice_request_media_type)
    target_format = _normalize_audio_format(config.voice_output_format)
    audio_filter = _build_ffmpeg_audio_filter(config)
    if source_format == target_format and not audio_filter:
        return None

    ffmpeg_path = (config.voice_ffmpeg_path or "ffmpeg").strip() or "ffmpeg"
    if "/" in ffmpeg_path or "\\" in ffmpeg_path:
        if not Path(ffmpeg_path).exists():
            return f"找不到 ffmpeg: {ffmpeg_path}"
        ffmpeg_executable = ffmpeg_path
    else:
        ffmpeg_executable = shutil.which(ffmpeg_path)
        if not ffmpeg_executable:
            return f"找不到 ffmpeg: {ffmpeg_path}"

    codec = (config.voice_ffmpeg_audio_codec or "").strip() if target_format == "amr" else ""
    if codec and not _ffmpeg_encoder_available(str(ffmpeg_executable), codec):
        return f"ffmpeg 不支持编码器: {codec}"
    return None


def _ffmpeg_encoder_available(ffmpeg_path: str, codec: str) -> bool:
    cache_key = (ffmpeg_path, codec)
    now = time.monotonic()
    cached = _ffmpeg_encoder_cache.get(cache_key)
    if cached and now - cached[0] < 60.0:
        return cached[1]

    try:
        result = subprocess.run(
            [ffmpeg_path, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
    except Exception as e:
        _ffmpeg_encoder_cache[cache_key] = (now, False, f"{type(e).__name__}: {e}")
        return False

    output = f"{result.stdout}\n{result.stderr}"
    available = result.returncode == 0 and codec in output
    _ffmpeg_encoder_cache[cache_key] = (now, available, output[:300])
    return available


async def is_voice_service_healthy(config: ScopedConfig, *, force: bool = False) -> bool:
    """Check GPT-SoVITS health with short timeout and TTL cache."""
    global _health_cache_key, _health_checked_at, _health_detail, _health_result

    if not _is_voice_configured(config):
        return False

    cache_key = ((config.voice_base_url or "").strip(), (config.voice_health_path or "").strip())
    now = time.monotonic()
    requirement_error = _conversion_requirement_detail(config) or _send_requirement_detail(config)
    if requirement_error:
        ttl = max(float(config.voice_unhealthy_cache_seconds or 0), 0.0)
        cache_is_valid = (
            not force
            and _health_cache_key == cache_key
            and _health_detail == requirement_error
            and now - _health_checked_at < ttl
        )
        if cache_is_valid:
            return False
        _health_cache_key = cache_key
        _health_checked_at = now
        _health_result = False
        _health_detail = requirement_error
        logger.warning(f"语音服务健康检查失败: {requirement_error}")
        return False

    ttl = config.voice_health_cache_seconds if _health_result else config.voice_unhealthy_cache_seconds
    ttl = max(float(ttl or 0), 0.0)

    if not force and _health_cache_key == cache_key and now - _health_checked_at < ttl:
        return _health_result

    async with _health_lock:
        now = time.monotonic()
        ttl = config.voice_health_cache_seconds if _health_result else config.voice_unhealthy_cache_seconds
        ttl = max(float(ttl or 0), 0.0)
        if not force and _health_cache_key == cache_key and now - _health_checked_at < ttl:
            return _health_result

        url = _build_url(config.voice_base_url, config.voice_health_path)
        had_cached_result = _health_cache_key == cache_key and _health_checked_at > 0
        old_result = _health_result
        try:
            timeout = httpx.Timeout(max(float(config.voice_health_timeout_seconds or 0), 0.1))
            async with httpx.AsyncClient(
                timeout=timeout,
                follow_redirects=True,
                trust_env=bool(config.voice_trust_env_proxy),
            ) as client:
                response = await client.get(url)
            healthy = response.status_code < 500
            detail = f"HTTP {response.status_code}"
        except Exception as e:
            healthy = False
            detail = f"{type(e).__name__}: {e}"

        _health_cache_key = cache_key
        _health_checked_at = time.monotonic()
        _health_result = healthy
        _health_detail = detail

        if not had_cached_result or healthy != old_result:
            if healthy:
                logger.info(f"语音服务健康检查通过: {url} ({detail})")
            else:
                logger.warning(f"语音服务健康检查失败: {url} ({detail})")
        return healthy


def get_voice_health_detail() -> str:
    return _health_detail


def _build_tts_payload(config: ScopedConfig, text: str) -> dict[str, object]:
    return {
        "text": text,
        "text_lang": config.voice_text_lang,
        "ref_audio_path": config.voice_ref_audio_path,
        "prompt_text": config.voice_prompt_text,
        "prompt_lang": config.voice_prompt_lang,
        "text_split_method": config.voice_text_split_method,
        "media_type": _normalize_audio_format(config.voice_request_media_type),
        "streaming_mode": int(config.voice_streaming_mode),
        "batch_size": int(config.voice_batch_size),
        "speed_factor": float(config.voice_speed_factor),
        "top_k": int(config.voice_top_k),
        "top_p": float(config.voice_top_p),
        "temperature": float(config.voice_temperature),
    }


async def _request_tts_audio(config: ScopedConfig, text: str) -> bytes:
    url = _build_url(config.voice_base_url, config.voice_tts_path)
    payload = _build_tts_payload(config, text)
    timeout = httpx.Timeout(max(float(config.voice_tts_timeout_seconds or 0), 1.0))

    async with _tts_lock:
        async with httpx.AsyncClient(timeout=timeout, trust_env=bool(config.voice_trust_env_proxy)) as client:
            response = await client.post(url, json=payload)

    if response.status_code != 200:
        error_text = response.text.replace("\n", " ").strip()
        raise VoiceToolError(f"TTS 接口返回 HTTP {response.status_code}: {error_text[:300]}")

    if not response.content:
        raise VoiceToolError("TTS 接口返回了空音频")

    return bytes(response.content)


def _convert_audio_sync(audio: bytes, source_format: str, target_format: str, config: ScopedConfig) -> bytes:
    source_format = _normalize_audio_format(source_format)
    target_format = _normalize_audio_format(target_format)
    audio_filter = _build_ffmpeg_audio_filter(config)
    if source_format == target_format and not audio_filter:
        return audio

    ffmpeg_path = (config.voice_ffmpeg_path or "ffmpeg").strip() or "ffmpeg"
    with tempfile.TemporaryDirectory(prefix="ai_groupmate_voice_") as temp_dir:
        workdir = Path(temp_dir)
        source_path = workdir / f"input.{source_format}"
        target_path = workdir / f"output.{target_format}"
        source_path.write_bytes(audio)

        command = [
            ffmpeg_path,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source_path),
        ]

        if audio_filter:
            command.extend(["-af", audio_filter])

        if target_format == "amr":
            command.extend(["-ar", str(int(config.voice_amr_sample_rate)), "-ac", "1"])
            if (config.voice_ffmpeg_audio_codec or "").strip():
                command.extend(["-c:a", config.voice_ffmpeg_audio_codec.strip()])
            if (config.voice_amr_bitrate or "").strip():
                command.extend(["-b:a", config.voice_amr_bitrate.strip()])

        command.append(str(target_path))

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=max(float(config.voice_ffmpeg_timeout_seconds or 0), 1.0),
                check=False,
            )
        except FileNotFoundError as e:
            raise VoiceToolError(f"找不到 ffmpeg: {ffmpeg_path}") from e
        except subprocess.TimeoutExpired as e:
            raise VoiceToolError("ffmpeg 转码超时") from e

        if result.returncode != 0:
            error_text = (result.stderr or result.stdout or "").replace("\n", " ").strip()
            raise VoiceToolError(f"ffmpeg 转码失败: {error_text[:300]}")

        if not target_path.exists() or target_path.stat().st_size <= 0:
            raise VoiceToolError("ffmpeg 转码没有生成有效音频")

        return target_path.read_bytes()


async def _convert_audio(audio: bytes, source_format: str, target_format: str, config: ScopedConfig) -> bytes:
    return await asyncio.to_thread(_convert_audio_sync, audio, source_format, target_format, config)


def _write_voice_file(audio: bytes, audio_format: str, config: ScopedConfig) -> Path:
    send_dir = (config.voice_send_dir or "").strip()
    if not send_dir:
        raise VoiceToolError("voice_send_method=path 时必须配置 voice_send_dir")

    directory = Path(send_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"ai_groupmate_voice_{int(time.time())}_{uuid.uuid4().hex}.{audio_format}"
    path.write_bytes(audio)
    return path.resolve()


async def _cleanup_voice_file_later(path: Path, delay_seconds: float) -> None:
    await asyncio.sleep(max(delay_seconds, 0.0))
    with contextlib.suppress(FileNotFoundError):
        path.unlink()


def _extract_message_id(result) -> str:
    if hasattr(result, "msg_ids"):
        msg_ids = getattr(result, "msg_ids", None) or []
        return str(msg_ids[-1]["message_id"]) if msg_ids else "unknown"
    if isinstance(result, dict):
        return str(result.get("message_id") or result.get("message_id_v2") or "unknown")
    return "unknown"


async def _send_voice_audio(audio: bytes, output_format: str, config: ScopedConfig, session_id: str):
    method = _normalize_voice_send_method(config.voice_send_method)
    if method == "raw":
        mimetype = _audio_mimetype(output_format)
        return await UniMessage.voice(raw=audio, mimetype=mimetype, name=f"voice.{output_format}").send()

    if method != "path":
        raise VoiceToolError(f"不支持的语音发送方式: {config.voice_send_method}")

    path = _write_voice_file(audio, output_format, config)
    try:
        try:
            group_id = int(session_id)
        except ValueError as e:
            raise VoiceToolError(f"path 发送方式只支持数字群号 session_id: {session_id}") from e
        record_file = path.as_uri()
        logger.info(f"准备发送语音文件: {record_file} size={path.stat().st_size} format={output_format}")
        bot = get_bot()
        message = Message(MessageSegment.record(record_file))
        return await bot.call_api("send_msg", message_type="group", group_id=group_id, message=message)
    finally:
        keep_seconds = max(float(config.voice_send_file_keep_seconds or 0), 0.0)
        if keep_seconds > 0:
            asyncio.create_task(_cleanup_voice_file_later(path, keep_seconds))
        else:
            with contextlib.suppress(FileNotFoundError):
                path.unlink()


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

    @tool("send_voice")
    async def send_voice(content: str) -> str:
        """
        将文本合成为语音并发送到当前群聊。
        只有用户明确要求你“发语音、用语音说、念出来、读出来”时才使用。

        Args:
            content: 要合成为语音的短文本。
        """
        if not await _is_current_request_active(session_id, request_id):
            return "请求已过期，已取消发送语音。"

        try:
            voice_text = _normalize_voice_text(content, int(config.voice_max_text_length))
            if not voice_text:
                return "语音文本为空，未发送。"

            if not await is_voice_service_healthy(config):
                return f"语音服务暂时不可用，未发送语音。健康状态: {get_voice_health_detail()}"

            audio = await _request_tts_audio(config, voice_text)

            if not await _is_current_request_active(session_id, request_id):
                return "请求已过期，已取消发送语音。"

            output_format = _normalize_audio_format(config.voice_output_format)
            audio = await _convert_audio(audio, config.voice_request_media_type, output_format, config)

            if not await _is_current_request_active(session_id, request_id):
                return "请求已过期，已取消发送语音。"

            result = await _send_voice_audio(audio, output_format, config, session_id)
            msg_id = _extract_message_id(result)
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
