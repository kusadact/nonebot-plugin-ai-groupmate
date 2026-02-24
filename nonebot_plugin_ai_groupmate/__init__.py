import random
import asyncio
import datetime
import traceback
import json
import base64
import shutil
import hashlib
import hmac
import urllib.error
import urllib.parse
import urllib.request
import threading
from io import BytesIO
from pathlib import Path
from typing import Any

import jieba
from PIL import Image as PILImage
from nonebot import logger, require, on_command, on_message, get_plugin_config
from nonebot.permission import SUPERUSER
from wordcloud import WordCloud
from nonebot.params import CommandArg
from nonebot.plugin import PluginMetadata, inherit_supported_adapters
from nonebot.typing import T_State
from nonebot.adapters import Bot, Event, Message

require("nonebot_plugin_alconna")
require("nonebot_plugin_orm")
require("nonebot_plugin_uninfo")
require("nonebot_plugin_localstore")
require("nonebot_plugin_apscheduler")
import nonebot_plugin_localstore as store
from sqlalchemy import Select
from sqlalchemy.exc import IntegrityError
from nonebot_plugin_orm import get_session, async_scoped_session
from nonebot_plugin_uninfo import Uninfo, SceneType, QryItrface
from nonebot_plugin_alconna import Image, UniMessage, image_fetch, get_message_id
from nonebot_plugin_apscheduler import scheduler
from nonebot_plugin_alconna.uniseg import UniMsg

from .vlm import image_vl
from .agent import choice_response_strategy
from .model import ChatHistory, MediaStorage, ChatHistorySchema
from .utils import (
    generate_file_hash,
    check_and_compress_image_bytes,
    process_and_vectorize_session_chats,
)
from .config import Config
from .milvus import MilvusOP

__plugin_meta__ = PluginMetadata(
    name="nonebot-plugin-ai-groupmate",
    description="AI虚拟群友",
    usage="@bot 让bot进行回复\n/词频 <统计天数>\n/群词频<统计天数>",
    type="application",
    homepage="https://github.com/kusadact/nonebot-plugin-ai-groupmate",
    config=Config,
    supported_adapters=inherit_supported_adapters("nonebot_plugin_alconna", "nonebot_plugin_uninfo"),
    extra={"author": "kusadact <959472968@qq.com>"},
)
plugin_data_dir: Path = store.get_plugin_data_dir()
pic_dir = plugin_data_dir / "pics"
pic_dir.mkdir(parents=True, exist_ok=True)
plugin_config = get_plugin_config(Config).ai_groupmate
with open(Path(__file__).parent / "stop_words.txt", encoding="utf-8") as f:
    stop_words = f.read().splitlines() + ["id", "回复"]

switch_file = plugin_data_dir / "switch.json"
_enabled = True
if switch_file.exists():
    try:
        _enabled = json.loads(switch_file.read_text(encoding="utf-8")).get("enabled", True)
    except Exception:
        _enabled = True

# Seedance 单并发闸门：账户同一时间仅允许一个生成任务
_seedance_busy_lock = threading.Lock()
_seedance_busy = False
_seedance_busy_request_id = ""
_seedance_busy_task_id = ""
_seedance_busy_owner = ""
_seedance_busy_session = ""
_seedance_busy_since: datetime.datetime | None = None


def _is_enabled() -> bool:
    return _enabled


def _set_enabled(value: bool) -> None:
    global _enabled
    _enabled = value
    switch_file.write_text(json.dumps({"enabled": value}, ensure_ascii=False), encoding="utf-8")


def _get_seedance_whitelist() -> set[str]:
    return {str(i).strip() for i in plugin_config.seedance_tool_whitelist if str(i).strip()}


def _try_acquire_seedance_slot(request_id: str, caller_id: str, session_id: str) -> tuple[bool, str]:
    global _seedance_busy, _seedance_busy_request_id, _seedance_busy_task_id
    global _seedance_busy_owner, _seedance_busy_session, _seedance_busy_since

    with _seedance_busy_lock:
        if _seedance_busy:
            since = _seedance_busy_since.strftime("%H:%M:%S") if _seedance_busy_since else "unknown"
            running = _seedance_busy_task_id or _seedance_busy_request_id or "unknown"
            msg = (
                "[Seedance] 当前已有任务正在生成中，请等待完成后再试，本次请求无效。\n"
                f"running_task={running}\n"
                f"started_at={since}"
            )
            return False, msg

        _seedance_busy = True
        _seedance_busy_request_id = request_id
        _seedance_busy_task_id = ""
        _seedance_busy_owner = caller_id
        _seedance_busy_session = session_id
        _seedance_busy_since = datetime.datetime.now()
        return True, ""


def _set_seedance_running_task_id(request_id: str, task_id: str) -> None:
    global _seedance_busy_task_id
    with _seedance_busy_lock:
        if _seedance_busy and _seedance_busy_request_id == request_id:
            _seedance_busy_task_id = task_id


def _release_seedance_slot(request_id: str) -> None:
    global _seedance_busy, _seedance_busy_request_id, _seedance_busy_task_id
    global _seedance_busy_owner, _seedance_busy_session, _seedance_busy_since
    with _seedance_busy_lock:
        if _seedance_busy and _seedance_busy_request_id == request_id:
            _seedance_busy = False
            _seedance_busy_request_id = ""
            _seedance_busy_task_id = ""
            _seedance_busy_owner = ""
            _seedance_busy_session = ""
            _seedance_busy_since = None


def _is_seedance_request(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False

    direct_hits = (
        "生图",
        "文生图",
        "出图",
        "画图",
        "画个",
        "画一下",
        "帮我画",
        "画一张",
        "来张",
        "来一张",
        "生成图片",
        "生成照片",
        "生成相片",
        "生成图像",
        "生成一张",
        "生成视频",
        "文生视频",
        "做视频",
        "视频生成",
    )
    if any(k in t for k in direct_hits):
        return True

    verbs = ("生成", "画", "做", "整", "来")
    media_words = ("图", "图片", "照片", "相片", "图像", "视频", "短片", "海报", "壁纸")
    return any(v in t for v in verbs) and any(m in t for m in media_words)


def _is_seedance_video_request(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    video_words = ("视频", "短片", "动图", "动画", "动起来", "做视频", "文生视频", "视频生成")
    return any(k in t for k in video_words)


def _event_at_bot(event: Event, bot: Bot) -> bool:
    # OneBot v11 fallback: directly inspect raw message segments.
    try:
        raw_msg = getattr(event, "message", None) or getattr(event, "original_message", None)
        if not raw_msg:
            return False

        for seg in raw_msg:
            seg_type = getattr(seg, "type", None)
            if seg_type != "at":
                continue
            data = getattr(seg, "data", None)
            qq = None
            if isinstance(data, dict):
                qq = data.get("qq") or data.get("target")
            if qq is None:
                qq = getattr(seg, "qq", None) or getattr(seg, "target", None)
            if qq is not None and str(qq).strip() == str(bot.self_id):
                return True
    except Exception:
        return False
    return False


def _extract_seedance_prompt(text: str) -> str:
    prompt = (text or "").strip()
    bot_name = (plugin_config.bot_name or "").strip()
    if bot_name and prompt.lower().startswith(bot_name.lower()):
        prompt = prompt[len(bot_name) :].lstrip(" :：，,")
    return prompt.strip()


async def _plain_text_mentions_bot(plain_text: str, bot: Bot, session: Uninfo, interface: QryItrface) -> bool:
    text = (plain_text or "").strip()
    if not text:
        return False
    if not (text.startswith("@") or text.startswith("＠")):
        return False

    bot_names: set[str] = set()
    configured = (plugin_config.bot_name or "").strip()
    if configured:
        bot_names.add(configured)

    try:
        members = await interface.get_members(SceneType.GROUP, session.scene.id)
    except Exception:
        members = []

    bot_id = str(bot.self_id)
    for member in members:
        if str(member.id) != bot_id:
            continue
        if member.user and member.user.name:
            bot_names.add(str(member.user.name).strip())
        if getattr(member, "nick", None):
            bot_names.add(str(member.nick).strip())
        break

    low_text = text.lower()
    for name in bot_names:
        if not name:
            continue
        n = name.lower()
        if low_text.startswith(f"@{n}") or low_text.startswith(f"＠{n}"):
            return True
    return False


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _hmac_sha256(key: bytes, text: str) -> bytes:
    return hmac.new(key, text.encode("utf-8"), hashlib.sha256).digest()


def _find_first_str(data: Any, keys: set[str]) -> str | None:
    lower_keys = {k.lower() for k in keys}
    if isinstance(data, dict):
        for k, v in data.items():
            if str(k).lower() in lower_keys and isinstance(v, (str, int, float)):
                return str(v)
            nested = _find_first_str(v, keys)
            if nested:
                return nested
    elif isinstance(data, list):
        for item in data:
            nested = _find_first_str(item, keys)
            if nested:
                return nested
    return None


def _collect_media_urls(data: Any) -> list[str]:
    urls: list[str] = []

    def walk(obj: Any) -> None:
        if isinstance(obj, str):
            if obj.startswith(("http://", "https://")):
                urls.append(obj)
            return
        if isinstance(obj, list):
            for item in obj:
                walk(item)
            return
        if isinstance(obj, dict):
            for k, v in obj.items():
                lk = str(k).lower()
                if lk in {"image_url", "video_url", "url"}:
                    walk(v)
                elif lk in {"image_urls", "video_urls", "urls"}:
                    walk(v)
                elif isinstance(v, (dict, list)):
                    walk(v)

    walk(data)
    dedup: list[str] = []
    seen: set[str] = set()
    for u in urls:
        if u not in seen:
            seen.add(u)
            dedup.append(u)
    return dedup


def _collect_base64_payloads(data: Any) -> list[str]:
    keys = {"binary_data_base64", "image_base64", "images_base64", "video_base64", "base64", "b64"}
    payloads: list[str] = []

    def add_payload(s: str) -> None:
        text = (s or "").strip()
        if not text:
            return
        if text.startswith("data:") and "," in text:
            text = text.split(",", 1)[1].strip()
        text = text.replace("\n", "").replace("\r", "")
        if len(text) < 64:
            return
        payloads.append(text)

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                lk = str(k).lower()
                if lk in keys:
                    if isinstance(v, str):
                        add_payload(v)
                    elif isinstance(v, list):
                        for item in v:
                            if isinstance(item, str):
                                add_payload(item)
                    continue
                if isinstance(v, (dict, list)):
                    walk(v)
            return
        if isinstance(obj, list):
            for item in obj:
                if isinstance(item, (dict, list)):
                    walk(item)

    walk(data)
    dedup: list[str] = []
    seen: set[str] = set()
    for p in payloads:
        if p not in seen:
            seen.add(p)
            dedup.append(p)
    return dedup


def _guess_generated_file_ext(raw: bytes, is_video: bool) -> str:
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if raw.startswith(b"\xff\xd8\xff"):
        return "jpg"
    if raw.startswith(b"GIF87a") or raw.startswith(b"GIF89a"):
        return "gif"
    if raw.startswith(b"RIFF") and len(raw) > 12 and raw[8:12] == b"WEBP":
        return "webp"
    if len(raw) > 12 and raw[4:8] == b"ftyp":
        return "mp4"
    return "mp4" if is_video else "png"


async def _save_seedance_base64_outputs(base64_payloads: list[str], request_id: str, is_video: bool) -> list[str]:
    static_dir = (plugin_config.seedance_static_dir or "").strip()
    static_base_url = (plugin_config.seedance_static_base_url or "").strip().rstrip("/")
    if not static_dir:
        raise RuntimeError("未配置 ai_groupmate__seedance_static_dir")
    if not static_base_url:
        raise RuntimeError("未配置 ai_groupmate__seedance_static_base_url")

    request_temp_dir = Path(static_dir) / "temp" / request_id
    request_temp_dir.mkdir(parents=True, exist_ok=True)

    urls: list[str] = []
    try:
        max_items = min(len(base64_payloads), 8)
        for idx, payload in enumerate(base64_payloads[:max_items], start=1):
            raw = base64.b64decode(payload)
            ext = _guess_generated_file_ext(raw, is_video=is_video)
            filename = f"result_{idx:02d}.{ext}"
            file_path = request_temp_dir / filename
            await asyncio.to_thread(file_path.write_bytes, raw)
            urls.append(f"{static_base_url}/temp/{request_id}/{filename}")
    finally:
        _schedule_seedance_temp_cleanup(request_temp_dir, request_id)

    logger.info(
        f"[SeedanceTemp] saved generated outputs request_id={request_id} "
        f"count={len(urls)} dir={request_temp_dir}"
    )
    return urls


def _volc_openapi_post(action: str, body: dict[str, Any], timeout: int = 30) -> dict[str, Any]:
    access_key_id = (plugin_config.seedance_access_key_id or "").strip()
    secret_access_key = (plugin_config.seedance_secret_access_key or "").strip()
    if not access_key_id or not secret_access_key:
        raise RuntimeError("未配置 ai_groupmate__seedance_access_key_id 或 ai_groupmate__seedance_secret_access_key")

    endpoint = (plugin_config.seedance_endpoint or "").strip() or "https://visual.volcengineapi.com"
    if "://" not in endpoint:
        endpoint = f"https://{endpoint}"

    parsed = urllib.parse.urlparse(endpoint)
    if not parsed.scheme or not parsed.netloc:
        raise RuntimeError(f"seedance_endpoint 非法: {endpoint}")

    host = parsed.netloc
    path = parsed.path or "/"
    if not path.startswith("/"):
        path = f"/{path}"

    query_pairs = [
        ("Action", action),
        ("Version", (plugin_config.seedance_api_version or "").strip() or "2022-08-31"),
    ]
    encoded_pairs = [
        (urllib.parse.quote(str(k), safe="-_.~"), urllib.parse.quote(str(v), safe="-_.~"))
        for k, v in query_pairs
    ]
    encoded_pairs.sort()
    canonical_query = "&".join([f"{k}={v}" for k, v in encoded_pairs])
    request_url = f"{parsed.scheme}://{host}{path}?{canonical_query}"

    payload = json.dumps(body, ensure_ascii=False, separators=(",", ":"))
    payload_hash = _sha256_hex(payload)
    x_date = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    short_date = x_date[:8]
    region = (plugin_config.seedance_region or "").strip() or "cn-north-1"
    service = (plugin_config.seedance_service or "").strip() or "cv"

    canonical_headers = (
        "content-type:application/json\n"
        f"host:{host}\n"
        f"x-content-sha256:{payload_hash}\n"
        f"x-date:{x_date}\n"
    )
    signed_headers = "content-type;host;x-content-sha256;x-date"
    canonical_request = "\n".join(
        [
            "POST",
            path,
            canonical_query,
            canonical_headers,
            signed_headers,
            payload_hash,
        ]
    )

    credential_scope = f"{short_date}/{region}/{service}/request"
    string_to_sign = "\n".join(
        [
            "HMAC-SHA256",
            x_date,
            credential_scope,
            _sha256_hex(canonical_request),
        ]
    )

    k_date = _hmac_sha256(secret_access_key.encode("utf-8"), short_date)
    k_region = _hmac_sha256(k_date, region)
    k_service = _hmac_sha256(k_region, service)
    k_signing = _hmac_sha256(k_service, "request")
    signature = hmac.new(k_signing, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()
    authorization = (
        "HMAC-SHA256 "
        f"Credential={access_key_id}/{credential_scope}, "
        f"SignedHeaders={signed_headers}, "
        f"Signature={signature}"
    )

    headers = {
        "Content-Type": "application/json",
        "Host": host,
        "X-Date": x_date,
        "X-Content-Sha256": payload_hash,
        "Authorization": authorization,
    }
    req = urllib.request.Request(
        url=request_url,
        data=payload.encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        err_raw = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code}: {err_raw}") from e
    except Exception as e:
        raise RuntimeError(f"请求火山接口失败: {e}") from e

    try:
        data = json.loads(raw) if raw else {}
    except Exception as e:
        raise RuntimeError(f"响应 JSON 解析失败: {e}; raw={raw[:300]}") from e

    if not isinstance(data, dict):
        raise RuntimeError(f"响应格式异常: {type(data).__name__}")

    # OpenAPI 典型错误结构
    response_meta = data.get("ResponseMetadata")
    if isinstance(response_meta, dict):
        err = response_meta.get("Error")
        if isinstance(err, dict):
            code = err.get("Code") or err.get("CodeN") or "UnknownError"
            msg = err.get("Message") or err.get("MessageCN") or str(err)
            req_id = response_meta.get("RequestId") or ""
            raise RuntimeError(f"{code}: {msg} request_id={req_id}".strip())

    # 兼容部分网关返回结构
    code = data.get("code")
    if isinstance(code, int) and code not in (0, 10000):
        message = data.get("message") or data.get("msg") or "unknown error"
        raise RuntimeError(f"code={code}, message={message}")

    return data


def _build_seedance_submit_payload(prompt: str, ref_urls: list[str], is_video: bool) -> tuple[str, dict[str, Any]]:
    if is_video:
        req_key = plugin_config.seedance_video_i2v_req_key if ref_urls else plugin_config.seedance_video_t2v_req_key
        model = (plugin_config.seedance_video_model or "").strip()
    else:
        req_key = plugin_config.seedance_image_i2i_req_key if ref_urls else plugin_config.seedance_image_t2i_req_key
        model = (plugin_config.seedance_image_model or "").strip()

    req_key = (req_key or "").strip()
    if not req_key:
        raise RuntimeError("缺少 req_key 配置，请检查 seedance_image_* 或 seedance_video_* 配置")

    payload: dict[str, Any] = {
        "req_key": req_key,
        "prompt": prompt,
    }
    if model:
        payload["model"] = model
    if ref_urls:
        payload["image_urls"] = ref_urls
    return req_key, payload


async def _seedance_submit_task(req_key: str, payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    action = (plugin_config.seedance_action_submit or "").strip() or "CVSync2AsyncSubmitTask"
    submit_resp = await asyncio.to_thread(_volc_openapi_post, action, payload)
    task_id = _find_first_str(submit_resp, {"task_id", "taskid", "TaskId"})
    if not task_id:
        raise RuntimeError(f"提交任务成功但未返回 task_id: {json.dumps(submit_resp, ensure_ascii=False)[:600]}")

    logger.info(
        f"[SeedanceAPI] submit ok req_key={req_key} task_id={task_id} "
        f"request_id={_find_first_str(submit_resp, {'request_id', 'RequestId'})}"
    )
    return task_id, submit_resp


async def _seedance_get_result_once(req_key: str, task_id: str) -> dict[str, Any]:
    action = (plugin_config.seedance_action_result or "").strip() or "CVSync2AsyncGetResult"
    payload = {"req_key": req_key, "task_id": task_id}
    return await asyncio.to_thread(_volc_openapi_post, action, payload)


async def _seedance_poll_result(req_key: str, task_id: str) -> dict[str, Any]:
    timeout = max(10, int(plugin_config.seedance_poll_timeout_seconds))
    interval = max(1, int(plugin_config.seedance_poll_interval_seconds))
    start = asyncio.get_running_loop().time()
    last_resp: dict[str, Any] = {}
    last_status = "unknown"
    success_status = {"done", "success", "succeeded", "finished", "complete", "completed"}
    failed_status = {"fail", "failed", "error", "cancel", "canceled", "cancelled"}

    while True:
        if asyncio.get_running_loop().time() - start > timeout:
            raise TimeoutError(
                f"任务轮询超时 {timeout}s, task_id={task_id}, last_status={last_status}, "
                f"last_resp={json.dumps(last_resp, ensure_ascii=False)[:400]}"
            )

        resp = await _seedance_get_result_once(req_key, task_id)
        last_resp = resp
        status_raw = _find_first_str(resp, {"status", "task_status", "state"}) or "unknown"
        last_status = status_raw
        status = status_raw.lower().strip()
        urls = _collect_media_urls(resp)

        if status in success_status:
            return resp
        if status in failed_status:
            raise RuntimeError(f"任务失败 status={status_raw}, resp={json.dumps(resp, ensure_ascii=False)[:500]}")
        if urls and status == "unknown":
            # 兼容极少数返回不带状态但已携带结果 URL 的场景
            return resp

        await asyncio.sleep(interval)


def _cleanup_seedance_temp_dir(temp_dir: str, request_id: str) -> None:
    target = Path(temp_dir)
    try:
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
            logger.info(f"[SeedanceTemp] cleaned request_id={request_id} dir={target}")
    except Exception as e:
        logger.error(f"[SeedanceTemp] cleanup failed request_id={request_id}: {e}")


def _cleanup_seedance_temp_expired_once() -> None:
    static_dir = (plugin_config.seedance_static_dir or "").strip()
    if not static_dir:
        return

    ttl_seconds = max(1, int(plugin_config.seedance_temp_ttl_minutes)) * 60
    base = Path(static_dir) / "temp"
    if not base.exists():
        return

    now_ts = datetime.datetime.now().timestamp()
    for child in base.iterdir():
        if not child.is_dir():
            continue
        try:
            age = now_ts - child.stat().st_mtime
            if age >= ttl_seconds:
                shutil.rmtree(child, ignore_errors=True)
                logger.info(f"[SeedanceTemp] gc removed expired dir={child} age_seconds={int(age)}")
        except Exception as e:
            logger.error(f"[SeedanceTemp] gc failed dir={child}: {e}")


def _schedule_seedance_temp_cleanup(temp_dir: Path, request_id: str) -> None:
    ttl = max(1, int(plugin_config.seedance_temp_ttl_minutes))
    run_date = datetime.datetime.now() + datetime.timedelta(minutes=ttl)
    job_id = f"seedance_cleanup_{request_id}"
    scheduler.add_job(
        _cleanup_seedance_temp_dir,
        trigger="date",
        run_date=run_date,
        id=job_id,
        replace_existing=True,
        kwargs={
            "temp_dir": str(temp_dir),
            "request_id": request_id,
        },
    )
    logger.info(f"[SeedanceTemp] scheduled cleanup request_id={request_id} ttl_minutes={ttl} dir={temp_dir}")


def _guess_image_ext(img: Image) -> str:
    value = str(getattr(img, "id", "") or "")
    value = value.split("?", 1)[0]
    if "." in value:
        ext = value.rsplit(".", 1)[-1].strip().lower()
        if ext.isalnum() and 1 <= len(ext) <= 6:
            return ext
    return "png"


async def _save_seedance_reference_images(
    imgs: list[Image],
    event: Event,
    bot: Bot,
    state: T_State,
    request_id: str,
) -> list[dict[str, str]]:
    max_refs = max(0, int(plugin_config.seedance_max_reference_images))
    if max_refs == 0:
        return []

    selected = list(imgs)[:max_refs]
    if not selected:
        return []

    static_dir = (plugin_config.seedance_static_dir or "").strip()
    static_base_url = (plugin_config.seedance_static_base_url or "").strip().rstrip("/")
    if not static_dir:
        raise RuntimeError("未配置 ai_groupmate__seedance_static_dir")
    if not static_base_url:
        raise RuntimeError("未配置 ai_groupmate__seedance_static_base_url")

    request_temp_dir = Path(static_dir) / "temp" / request_id
    request_temp_dir.mkdir(parents=True, exist_ok=True)

    refs: list[dict[str, str]] = []
    try:
        for idx, img in enumerate(selected, start=1):
            ext = _guess_image_ext(img)
            filename = f"{idx:02d}.{ext}"
            file_path = request_temp_dir / filename

            image_bytes = await asyncio.wait_for(image_fetch(event, bot, state, img), timeout=20.0)
            await asyncio.to_thread(file_path.write_bytes, image_bytes)

            public_url = f"{static_base_url}/temp/{request_id}/{filename}"
            refs.append(
                {
                    "index": str(idx),
                    "filename": filename,
                    "url": public_url,
                }
            )

        logger.info(
            f"[SeedanceTemp] saved request_id={request_id} count={len(refs)} "
            f"dir={request_temp_dir} order={[r['filename'] for r in refs]}"
        )
        return refs
    finally:
        _schedule_seedance_temp_cleanup(request_temp_dir, request_id)


async def _send_bot_reply_and_record(db_session: async_scoped_session, session_id: str, content: str) -> None:
    res = await UniMessage.text(content).send()
    msg_id = res.msg_ids[-1]["message_id"] if res.msg_ids else "unknown"
    chat_history = ChatHistory(
        session_id=session_id,
        user_id=plugin_config.bot_name,
        content_type="bot",
        content=f"id:{msg_id}\n" + content,
        user_name=plugin_config.bot_name,
    )
    db_session.add(chat_history)
    await db_session.commit()
    logger.info(f"Bot已回复: {content}")


def _map_static_url_to_local_path(url: str) -> Path | None:
    static_base_url = (plugin_config.seedance_static_base_url or "").strip().rstrip("/")
    static_dir = (plugin_config.seedance_static_dir or "").strip()
    if not static_base_url or not static_dir:
        return None

    try:
        base = urllib.parse.urlparse(static_base_url)
        target = urllib.parse.urlparse(url)
    except Exception:
        return None

    if not target.scheme or not target.netloc:
        return None
    if (target.scheme, target.netloc) != (base.scheme, base.netloc):
        return None

    base_path = base.path.rstrip("/")
    target_path = target.path
    if base_path and not target_path.startswith(base_path + "/"):
        return None

    rel = target_path[len(base_path) :].lstrip("/") if base_path else target_path.lstrip("/")
    if not rel:
        return None
    return Path(static_dir).joinpath(*rel.split("/"))


def _download_url_bytes(url: str, timeout: int = 20) -> bytes:
    req = urllib.request.Request(url=url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


async def _send_seedance_images_and_record(db_session: async_scoped_session, session_id: str, image_urls: list[str]) -> tuple[int, int]:
    success = 0
    failed = 0
    for idx, url in enumerate(image_urls, start=1):
        try:
            sent_res = None
            local_path = _map_static_url_to_local_path(url)
            if local_path and local_path.exists():
                image_bytes = await asyncio.to_thread(local_path.read_bytes)
                sent_res = await UniMessage.image(raw=image_bytes).send()
            else:
                try:
                    sent_res = await UniMessage.image(url=url).send()
                except Exception:
                    image_bytes = await asyncio.to_thread(_download_url_bytes, url, 20)
                    sent_res = await UniMessage.image(raw=image_bytes).send()

            msg_id = sent_res.msg_ids[-1]["message_id"] if sent_res and sent_res.msg_ids else "unknown"
            chat_history = ChatHistory(
                session_id=session_id,
                user_id=plugin_config.bot_name,
                content_type="bot",
                content=f"id:{msg_id}\n[Seedance] 发送生成图片 第{idx}张",
                user_name=plugin_config.bot_name,
            )
            db_session.add(chat_history)
            await db_session.commit()
            success += 1
        except Exception as e:
            failed += 1
            await db_session.rollback()
            logger.error(f"[Seedance] 发送生成图片失败 idx={idx} url={url}: {type(e).__name__}: {e}")
    return success, failed


async def _send_seedance_videos_and_record(db_session: async_scoped_session, session_id: str, video_urls: list[str]) -> tuple[int, int]:
    success = 0
    failed = 0
    for idx, url in enumerate(video_urls, start=1):
        try:
            sent_res = None
            local_path = _map_static_url_to_local_path(url)
            if local_path and local_path.exists():
                video_bytes = await asyncio.to_thread(local_path.read_bytes)
                if hasattr(UniMessage, "video"):
                    sent_res = await UniMessage.video(raw=video_bytes).send()
                else:
                    raise RuntimeError("当前 UniMessage 不支持 video 发送")
            else:
                if hasattr(UniMessage, "video"):
                    try:
                        sent_res = await UniMessage.video(url=url).send()
                    except Exception:
                        video_bytes = await asyncio.to_thread(_download_url_bytes, url, 30)
                        sent_res = await UniMessage.video(raw=video_bytes).send()
                else:
                    raise RuntimeError("当前 UniMessage 不支持 video 发送")

            msg_id = sent_res.msg_ids[-1]["message_id"] if sent_res and sent_res.msg_ids else "unknown"
            chat_history = ChatHistory(
                session_id=session_id,
                user_id=plugin_config.bot_name,
                content_type="bot",
                content=f"id:{msg_id}\n[Seedance] 发送生成视频 第{idx}条",
                user_name=plugin_config.bot_name,
            )
            db_session.add(chat_history)
            await db_session.commit()
            success += 1
        except Exception as e:
            failed += 1
            await db_session.rollback()
            logger.error(f"[Seedance] 发送生成视频失败 idx={idx} url={url}: {type(e).__name__}: {e}")
    return success, failed


async def _handle_seedance_request_direct(
    db_session: async_scoped_session,
    session: Uninfo,
    plain_text: str,
    to_me: bool,
    imgs: list[Image],
    event: Event,
    bot: Bot,
    state: T_State,
) -> bool:
    # 返回 True 表示该消息已被 Seedance 分支处理（含静默忽略）
    logger.debug(
        f"[SeedanceGate] probe caller={session.user.id} session_id={session.scene.id} "
        f"to_me={to_me} text={plain_text[:120]!r}"
    )
    if not _is_seedance_request(plain_text):
        return False

    caller_id = str(session.user.id)
    whitelist = _get_seedance_whitelist()
    request_id = f"seedance_gate_{int(datetime.datetime.now().timestamp())}_{random.randint(1000, 9999)}"

    # 非白名单：直接忽略，不回复
    if caller_id not in whitelist:
        logger.warning(
            f"[SeedanceGate] request_id={request_id} auth_passed=false "
            f"caller={caller_id} session_id={session.scene.id} action=ignore"
        )
        return True

    # 白名单但未明确对 bot 发起请求：也忽略，避免误触发
    if not to_me:
        logger.info(
            f"[SeedanceGate] request_id={request_id} auth_passed=true "
            f"caller={caller_id} session_id={session.scene.id} action=ignore_not_to_me"
        )
        return True

    acquired, busy_msg = _try_acquire_seedance_slot(request_id, caller_id, session.scene.id)
    if not acquired:
        logger.warning(
            f"[SeedanceGate] request_id={request_id} auth_passed=true "
            f"caller={caller_id} session_id={session.scene.id} action=busy_reject"
        )
        try:
            await _send_bot_reply_and_record(db_session, session.scene.id, busy_msg)
        except Exception:
            logger.exception(f"[SeedanceGate] request_id={request_id} busy_reply_failed")
        return True

    # 白名单 + 明确请求：固定路径处理，避免 agent 幻觉
    logger.success(
        f"[SeedanceGate] request_id={request_id} auth_passed=true "
        f"caller={caller_id} session_id={session.scene.id} action=process"
    )
    try:
        ref_images = await _save_seedance_reference_images(imgs, event, bot, state, request_id)
        ref_urls = [r["url"] for r in ref_images]
        is_video = _is_seedance_video_request(plain_text)
        prompt = _extract_seedance_prompt(plain_text)
        if not prompt:
            prompt = "请生成一段短视频" if is_video else "请生成一张图片"

        req_key, submit_payload = _build_seedance_submit_payload(prompt, ref_urls, is_video)
        task_id, _ = await _seedance_submit_task(req_key, submit_payload)
        _set_seedance_running_task_id(request_id, task_id)
        await _send_bot_reply_and_record(
            db_session,
            session.scene.id,
            (
                f"[Seedance] request_id={request_id} 任务已提交。\n"
                f"task_id={task_id}\n"
                f"类型={'视频' if is_video else '图片'} req_key={req_key}\n"
                f"状态=排队/生成中，请稍候..."
            ),
        )
        result_resp = await _seedance_poll_result(req_key, task_id)
        result_urls = _collect_media_urls(result_resp)
        if ref_urls:
            ref_set = set(ref_urls)
            result_urls = [u for u in result_urls if u not in ref_set]
        base64_hint = ""
        if not result_urls:
            base64_payloads = _collect_base64_payloads(result_resp)
            if base64_payloads:
                try:
                    result_urls = await _save_seedance_base64_outputs(base64_payloads, request_id, is_video)
                except Exception as e:
                    base64_hint = (
                        f"\n检测到 {len(base64_payloads)} 个base64结果，"
                        f"但落盘失败: {type(e).__name__}: {e}"
                    )
        media_name = "视频" if is_video else "图片"

        if ref_images:
            ref_desc = f"参考图数量={len(ref_images)}（顺序按用户发送）\n"
        else:
            ref_desc = "未检测到参考图（本次仅文本提示词）\n"

        if result_urls:
            result_lines = "\n".join([f"{idx}. {u}" for idx, u in enumerate(result_urls, start=1)])
            if not is_video:
                reply_text = (
                    f"[Seedance] request_id={request_id} 任务成功。\n"
                    f"task_id={task_id}\n"
                    f"类型={media_name} req_key={req_key}\n"
                    f"{ref_desc}"
                    f"{media_name}结果URL：\n{result_lines}\n"
                    f"已生成 {len(result_urls)} 张图片，正在发送。\n"
                    f"临时文件将在 {max(1, int(plugin_config.seedance_temp_ttl_minutes))} 分钟后自动清理。"
                )
                await _send_bot_reply_and_record(
                    db_session,
                    session.scene.id,
                    reply_text,
                )
                sent_cnt, fail_cnt = await _send_seedance_images_and_record(db_session, session.scene.id, result_urls)
                if fail_cnt > 0:
                    await _send_bot_reply_and_record(
                        db_session,
                        session.scene.id,
                        f"[Seedance] 图片发送完成：成功 {sent_cnt}/{len(result_urls)}，失败 {fail_cnt}。",
                    )
                return True

            reply_text = (
                f"[Seedance] request_id={request_id} 任务成功。\n"
                f"task_id={task_id}\n"
                f"类型={media_name} req_key={req_key}\n"
                f"{ref_desc}"
                f"{media_name}结果URL：\n{result_lines}\n"
                f"已生成 {len(result_urls)} 条视频，正在发送。\n"
                f"临时文件将在 {max(1, int(plugin_config.seedance_temp_ttl_minutes))} 分钟后自动清理。"
            )
            await _send_bot_reply_and_record(
                db_session,
                session.scene.id,
                reply_text,
            )
            sent_cnt, fail_cnt = await _send_seedance_videos_and_record(db_session, session.scene.id, result_urls)
            if fail_cnt > 0:
                await _send_bot_reply_and_record(
                    db_session,
                    session.scene.id,
                    f"[Seedance] 视频发送完成：成功 {sent_cnt}/{len(result_urls)}，失败 {fail_cnt}。",
                )
            return True
        else:
            status = _find_first_str(result_resp, {"status", "task_status", "state"}) or "unknown"
            reply_text = (
                f"[Seedance] request_id={request_id} 任务完成但未解析到{media_name}URL。\n"
                f"task_id={task_id} req_key={req_key} status={status}\n"
                f"{ref_desc}"
                f"原始返回片段：{json.dumps(result_resp, ensure_ascii=False)[:500]}"
                f"{base64_hint}"
            )

        await _send_bot_reply_and_record(
            db_session,
            session.scene.id,
            reply_text,
        )
    except Exception as e:
        err_msg = f"{type(e).__name__}: {e}"
        if "HTTP 401" in str(e) and "Access Denied" in str(e):
            if ref_urls and not is_video:
                err_text = (
                    f"[Seedance] request_id={request_id} 调用失败：图生图权限被拒绝（401 Access Denied）。\n"
                    f"当前使用 req_key={req_key}。\n"
                    f"请检查 ai_groupmate__seedance_image_i2i_req_key 是否为你账号已开通的图生图模型，"
                    f"或在火山控制台开通对应能力。"
                )
            elif ref_urls and is_video:
                err_text = (
                    f"[Seedance] request_id={request_id} 调用失败：图生视频权限被拒绝（401 Access Denied）。\n"
                    f"当前使用 req_key={req_key}。\n"
                    f"请检查 ai_groupmate__seedance_video_i2v_req_key 是否为你账号已开通的模型。"
                )
            else:
                err_text = (
                    f"[Seedance] request_id={request_id} 调用失败：接口鉴权被拒绝（401 Access Denied）。\n"
                    f"请检查 AK/SK、Region、Service 与账号权限是否匹配。"
                )
        else:
            err_text = f"[Seedance] request_id={request_id} 调用失败: {err_msg}"
        logger.exception(err_text)
        try:
            await _send_bot_reply_and_record(db_session, session.scene.id, err_text)
        except Exception:
            logger.exception(f"[SeedanceGate] request_id={request_id} error_reply_failed")
    finally:
        _release_seedance_slot(request_id)
    return True


ai = on_command("ai", permission=SUPERUSER)


@ai.handle()
async def _(arg: Message = CommandArg()):
    sub = arg.extract_plain_text().strip().lower()

    if sub in {"on", "enable", "start", "1", "true", "开", "开启", "启用"}:
        _set_enabled(True)
        await ai.finish("ai enabled")
    elif sub in {"off", "disable", "stop", "0", "false", "关", "关闭", "停用"}:
        _set_enabled(False)
        await ai.finish("ai disabled")
    elif sub in {"status", "state", "s", "", "状态"}:
        status = "on" if _is_enabled() else "off"
        milvus_uri = (plugin_config.milvus_uri or "").strip()
        milvus_line = "n/a"

        if milvus_uri:
            is_remote = ("://" in milvus_uri) or (milvus_uri.count(":") == 1 and milvus_uri.rsplit(":", 1)[1].isdigit())
            if not is_remote:
                # Milvus Lite / local file path
                p = Path(milvus_uri)
                milvus_line = "ok" if p.exists() else "down (local)"
            else:
                # TCP probe first: distinguish "tunnel down" vs "Milvus down" behind the tunnel.
                host = None
                port = None
                try:
                    if "://" in milvus_uri:
                        from urllib.parse import urlparse

                        u = urlparse(milvus_uri)
                        host = u.hostname
                        port = u.port or 19530
                    else:
                        host, port_s = milvus_uri.rsplit(":", 1)
                        port = int(port_s)
                except Exception:
                    host = None
                    port = None

                tunnel_ok: bool | None = None
                tunnel_err = ""
                if host and port:
                    try:
                        r, w = await asyncio.wait_for(asyncio.open_connection(host, port), timeout=0.8)
                        w.close()
                        try:
                            await w.wait_closed()
                        except Exception:
                            pass
                        tunnel_ok = True
                    except asyncio.TimeoutError:
                        tunnel_ok = False
                        tunnel_err = "timeout"
                    except Exception as e:
                        tunnel_ok = False
                        tunnel_err = type(e).__name__

                if tunnel_ok is False:
                    milvus_line = f"down (tunnel:{tunnel_err})" if tunnel_err else "down (tunnel)"
                else:
                    try:
                        client = MilvusOP._get_async_client()
                        try:
                            await asyncio.wait_for(client.list_collections(), timeout=2.0)
                        except AttributeError:
                            await asyncio.wait_for(client.has_collection(collection_name="chat_collection"), timeout=2.0)
                        milvus_line = "ok"
                    except asyncio.TimeoutError:
                        milvus_line = "down (milvus:timeout)"
                    except Exception as e:
                        milvus_line = f"down (milvus:{type(e).__name__})"

        await ai.finish(f"ai status: {status}\nmilvus: {milvus_line}")
    else:
        await ai.finish("Usage: /ai on|off|status")


record = on_message(
    priority=999,
    block=True,
)


@record.handle()
async def handle_message(db_session: async_scoped_session, msg: UniMsg, session: Uninfo, event: Event, bot: Bot, state: T_State, interface: QryItrface):
    """处理消息的主函数"""
    imgs = msg.include(Image)
    content = f"id: {get_message_id()}\n"
    to_me = False
    is_text = False
    if event.is_tome() or _event_at_bot(event, bot):
        to_me = True
        content += f"@{plugin_config.bot_name} "

    at_targets: list[str] = []
    for i in msg:
        if i.type == "at":
            target_id = str(getattr(i, "target", "") or "").strip()
            if target_id:
                at_targets.append(target_id)
            if target_id and target_id == str(bot.self_id):
                to_me = True
            members = await interface.get_members(SceneType.GROUP, session.scene.id)
            for member in members:
                if member.id == i.target:
                    name = member.user.name if member.user.name else ""
                    break
            else:
                continue
            content += "@" + name + " "
            is_text = True
        if i.type == "reply":
            content += "回复id:" + i.id
        if i.type == "text":
            content += i.text
            is_text = True

    # 构建用户名（包含昵称和职位）
    user_name = session.user.name
    if session.member:
        if session.member.nick:
            user_name = f"({session.member.nick}){user_name}"
        if session.member.role:
            if session.member.role.name == "owner":
                user_name = f"群主-{user_name}"
            elif session.member.role.name == "admin":
                user_name = f"管理员-{user_name}"

    # ========== 步骤1: 处理文本消息（快速） ==========
    if is_text:
        chat_history = ChatHistory(
            session_id=session.scene.id,
            user_id=session.user.id,
            content_type="text",
            content=content,
            user_name=user_name,
        )
        db_session.add(chat_history)

    # 立即提交文本消息
    try:
        await db_session.commit()
    except Exception as e:
        logger.error(f"保存文本消息失败: {e}")
        await db_session.rollback()

    # ========== 步骤2: 处理图片消息（耗时） ==========
    for img in imgs:
        await process_image_message(db_session, img, event, bot, state, session, user_name, f"id: {get_message_id()}\n")

    if not _is_enabled():
        return

    # ========== 步骤3: 决定是否回复 ==========
    plain_text = msg.extract_plain_text().strip()
    plain_event_text = event.get_plaintext() or ""

    bot_name_l = (plugin_config.bot_name or "").strip().lower()
    if bot_name_l and (
        plain_text.lower().startswith(bot_name_l)
        or plain_text.lower().startswith(f"@{bot_name_l}")
    ):
        to_me = True
    elif not to_me and await _plain_text_mentions_bot(plain_text, bot, session, interface):
        to_me = True

    logger.debug(
        f"[ToMeCheck] event.is_tome={event.is_tome()} bot.self_id={bot.self_id} "
        f"at_targets={at_targets} bot_name={plugin_config.bot_name!r} to_me={to_me}"
    )

    # Seedance 请求走固定分支：非白名单静默忽略，白名单失败直出报错，不交给 agent 幻觉发挥
    if await _handle_seedance_request_direct(db_session, session, plain_text, to_me, imgs, event, bot, state):
        return

    should_reply = to_me or (random.random() < plugin_config.reply_probability)
    if not plain_event_text and not imgs:
        should_reply = False
    if plain_event_text.startswith(("!", "！", "/", "#", "?", "\\")):
        should_reply = False
    if not plain_event_text and not event.is_tome():
        should_reply = False
    if to_me:
        user_id = session.user.id
        user_name = session.user.name or session.user.nick
    else:
        user_id = ""
        user_name = ""
    if should_reply:
        await handle_reply_logic(db_session, session, user_id, user_name)

    await db_session.commit()


async def process_image_message(
    db_session,
    img: Image,
    event: Event,
    bot: Bot,
    state: T_State,
    session: Uninfo,
    user_name: str | None,
    content: str,
):
    """处理单张图片消息"""
    content_type = "image"
    if not img.id:
        return
    image_format = img.id.split(".")[-1]

    # 获取和压缩图片
    pic = await asyncio.wait_for(image_fetch(event, bot, state, img), timeout=15.0)
    pic = await asyncio.to_thread(check_and_compress_image_bytes, pic, image_format=image_format.upper())
    file_hash = generate_file_hash(pic)
    file_path = pic_dir / f"{file_hash}.{image_format}"

    # 保存文件
    if not file_path.exists():
        file_path.write_bytes(pic)
    try:
        # 查询或创建媒体记录
        existing_media = (await db_session.execute(Select(MediaStorage).where(MediaStorage.file_hash == file_hash))).scalar()

        if existing_media:
            # 已存在，直接使用描述
            image_description = existing_media.description
            existing_media.references += 1
            db_session.add(existing_media)
        else:
            # 新图片，调用VLM获取描述
            image_description = await image_vl(file_path)

            if image_description:
                media_storage = MediaStorage(
                    file_hash=file_hash,
                    file_path=f"{file_hash}.{image_format}",
                    references=1,
                    description=image_description,
                )
                db_session.add(media_storage)
                await db_session.flush()  # 确保获取media_id
                existing_media = media_storage
            else:
                file_path.unlink()

        # 添加聊天历史记录
        if existing_media and image_description:
            chat_history = ChatHistory(
                session_id=session.scene.id,
                user_id=session.user.id,
                content_type=content_type,
                content=content + image_description,
                user_name=user_name or "",
                media_id=existing_media.media_id,
            )
            db_session.add(chat_history)

        await db_session.commit()

    except IntegrityError:
        # 处理并发插入冲突
        await db_session.rollback()
        existing_media = (await db_session.execute(Select(MediaStorage).where(MediaStorage.file_hash == file_hash))).scalar()

        if existing_media:
            existing_media.references += 1
            chat_history = ChatHistory(
                session_id=session.scene.id,
                user_id=session.user.id,
                content_type=content_type,
                content=content + existing_media.description,
                user_name=user_name or "",
                media_id=existing_media.media_id,
            )
            db_session.add(chat_history)
        await db_session.commit()

    except Exception as e:
        logger.error(f"处理图片失败: {e}")
        await db_session.rollback()


async def handle_reply_logic(
    db_session,
    session: Uninfo,
    user_id: str,
    user_name: str | None,
):
    """处理回复逻辑"""
    try:
        # 获取最近1小时内的消息历史
        cutoff_time = datetime.datetime.now() - datetime.timedelta(hours=1)
        last_msg = (await db_session.execute(Select(ChatHistory).where(ChatHistory.session_id == session.scene.id).where(ChatHistory.created_at >= cutoff_time).order_by(ChatHistory.msg_id.desc()).limit(20))).scalars().all()

        if not last_msg:
            logger.info("没有历史消息，跳过回复")
            return

        # 转换为模型对象并反转顺序（从旧到新）
        last_msg = [ChatHistorySchema.model_validate(m) for m in last_msg]
        last_msg = last_msg[::-1]

        # 使用Agent决定回复策略
        logger.info("开始调用Agent决策...")
        try:
            await asyncio.wait_for(choice_response_strategy(db_session, session.scene.id, last_msg, user_id, user_name, ""),timeout=120.0)
        except asyncio.TimeoutError:
            logger.warning(f"Agent 思考超时，跳过回复 - session: {session.scene.id}")
            return

    except Exception as e:
        logger.error(f"回复逻辑执行失败: {e}")
        await db_session.rollback()


def _build_wordcloud_image(words: str) -> BytesIO:
    """Generate a PNG image bytes object from words using WordCloud."""
    wc = WordCloud(font_path=Path(__file__).parent / "SourceHanSans.otf", width=1000, height=500).generate(words).to_image()
    image_bytes = BytesIO()
    wc.save(image_bytes, format="PNG")
    image_bytes.seek(0)
    return image_bytes


async def _collect_words_from_db(db_session, session_id: str, days: int = 1, user_id: str | None = None) -> str:
    """Query chat history and return a cleaned space-joined word string for wordcloud."""
    cutoff = datetime.datetime.now() - datetime.timedelta(days=days)
    where = [ChatHistory.session_id == session_id, ChatHistory.content_type == "text", ChatHistory.created_at >= cutoff]
    if user_id:
        where.append(ChatHistory.user_id == user_id)

    res = await db_session.execute(Select(ChatHistory.content).where(*where))
    ans = res.scalars().all()
    # tokenize and join
    ans = [" ".join([j.strip() for j in jieba.lcut(i)]) for i in ans]
    words = " ".join(ans)
    for sw in stop_words:
        words = words.replace(sw, "")
    return words


frequency = on_command("词频")


@frequency.handle()
async def _(db_session: async_scoped_session, session: Uninfo, arg: Message = CommandArg()):
    if not _is_enabled():
        await frequency.finish("ai_groupmate disabled")
    session_id = session.scene.id
    arg_text = arg.extract_plain_text().strip()
    if not arg_text:
        arg_text = "1"
    if not arg_text.isdigit():
        await frequency.finish("统计范围应为纯数字")
    days = int(arg_text)

    words = await _collect_words_from_db(db_session, session_id, days=days, user_id=session.user.id)
    if not words:
        await frequency.finish("在指定时间内，没有说过话呢")

    image_bytes = _build_wordcloud_image(words)
    await UniMessage.image(raw=image_bytes).send(reply_to=True)


group_frequency = on_command("群词频")


@group_frequency.handle()
async def _(db_session: async_scoped_session, session: Uninfo, arg: Message = CommandArg()):
    if not _is_enabled():
        await group_frequency.finish("ai_groupmate disabled")
    session_id = session.scene.id
    arg_text = arg.extract_plain_text().strip()
    if not arg_text:
        arg_text = "1"
    if not arg_text.isdigit():
        await group_frequency.finish("统计范围应为纯数字")
    days = int(arg_text)

    words = await _collect_words_from_db(db_session, session_id, days=days, user_id=None)
    # Even if no words, return an empty wordcloud (original group_frequency didn't check emptiness)
    if not words:
        await group_frequency.finish("在指定时间内，没有消息可统计")

    image_bytes = _build_wordcloud_image(words)
    await UniMessage.image(raw=image_bytes).send(reply_to=True)


@scheduler.scheduled_job("interval", minutes=60, max_instances=1, coalesce=True, id="vectorize_chat")
async def vectorize_message_history():
    if not _is_enabled():
        return
    async with get_session() as db_session:
        session_ids = await db_session.execute(Select(ChatHistory.session_id.distinct()))
        session_ids = session_ids.scalars().all()
        logger.info("开始向量化会话")
        for session_id in session_ids:
            try:
                res = await process_and_vectorize_session_chats(db_session, session_id)
                if res:
                    logger.info(f"向量化会话 {res['session_id']} 成功，共处理 {res['processed_groups']}/{res['total_groups']} 组")
                else:
                    logger.info(f"{session_id} 无需向量化")
            except Exception as e:
                print(traceback.format_exc())
                logger.error(f"向量化会话 {session_id} 失败: {e}")
                continue


@scheduler.scheduled_job("interval", minutes=10, max_instances=1, coalesce=True, id="seedance_temp_gc")
async def seedance_temp_gc_job():
    _cleanup_seedance_temp_expired_once()


@scheduler.scheduled_job("interval", minutes=30, max_instances=1, coalesce=True, id="vectorize_media")
async def vectorize_media():
    if not _is_enabled():
        return
    async with get_session() as db_session:
        medias_res = await db_session.execute(Select(MediaStorage).where(MediaStorage.references >= 3, MediaStorage.vectorized.is_(False)))
        medias = medias_res.scalars().all()
        logger.info(f"待向量化媒体数量: {len(medias)}")

        for media in medias:
            try:
                file_path = pic_dir / media.file_path
                if not file_path.exists():
                    logger.warning(f"文件不存在: {file_path}")
                    media.vectorized = True
                    db_session.add(media)
                    continue

                # 判断是否适合作为表情包
                vlm_res = await image_vl(file_path, "请判断这张图适不适合作为表情包，只回答是或否")
                if not vlm_res or vlm_res != "是":
                    media.vectorized = True
                    db_session.add(media)
                    continue

                try:
                    await MilvusOP.insert_media(media.media_id, [str(file_path)])
                    media.vectorized = True
                    db_session.add(media)
                    logger.info("向量化成功")
                except Exception as e:
                    logger.error(f"向量化插入失败 {media.media_id}: {e}")
                    # don't mark as vectorized so it can retry later
                    continue

            except Exception as e:
                logger.error(f"处理媒体 {getattr(media, 'media_id', 'unknown')} 失败: {e}")
                continue

        await db_session.commit()
        logger.info("向量化媒体完成")


@scheduler.scheduled_job("interval", minutes=35, max_instances=1, coalesce=True, id="clear_cache")
async def clear_cache_pic():
    async with get_session() as db_session:
        result = await db_session.execute(Select(MediaStorage).where(MediaStorage.references < 3, datetime.datetime.now() - MediaStorage.created_at > datetime.timedelta(days=15)))
        medias = result.scalars().all()

        if not medias:
            logger.info("没有需要清理的媒体文件")
            return

        records_to_delete = []
        for media in medias:
            try:
                file_path = Path(pic_dir / media.file_path)
                # use pathlib unlink with missing_ok=True to avoid raising if missing
                await asyncio.to_thread(file_path.unlink, True)
                records_to_delete.append(media)
                logger.debug(f"删除文件: {file_path}")
            except Exception as e:
                logger.error(f"删除文件失败 {getattr(media, 'file_path', 'unknown')}: {e}")
                records_to_delete.append(media)

        for media in records_to_delete:
            try:
                await db_session.delete(media)
            except Exception as e:
                logger.error(f"删除数据库记录失败 {getattr(media, 'media_id', 'unknown')}: {e}")

        await db_session.commit()
        logger.info(f"成功清理 {len(records_to_delete)} 个媒体记录")
