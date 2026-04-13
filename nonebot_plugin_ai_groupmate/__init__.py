import random
import asyncio
import datetime
import traceback
import json
import re
import base64
import mimetypes
from io import BytesIO
from pathlib import Path
from typing import Any

import jieba
from nonebot import logger, require, on_command, on_message, get_plugin_config, get_driver
from nonebot.permission import SUPERUSER
from pydantic import SecretStr
from wordcloud import WordCloud
from nonebot.params import CommandArg
from nonebot.plugin import PluginMetadata, inherit_supported_adapters
from nonebot.typing import T_State
from nonebot.adapters import Bot, Event, Message
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage as LCHumanMessage, SystemMessage as LCSystemMessage

require("nonebot_plugin_alconna")
require("nonebot_plugin_orm")
require("nonebot_plugin_uninfo")
require("nonebot_plugin_localstore")
require("nonebot_plugin_apscheduler")
import nonebot_plugin_localstore as store
from sqlalchemy import Select, desc, func as sqlfunc
from nonebot_plugin_orm import get_session, async_scoped_session
from nonebot_plugin_uninfo import Uninfo, SceneType, QryItrface
from nonebot_plugin_alconna import Image, UniMessage, image_fetch, get_message_id
from nonebot_plugin_apscheduler import scheduler
from nonebot_plugin_alconna.uniseg import UniMsg

from .agent import check_if_should_reply, choice_response_strategy
from .model import ChatHistory, MediaStorage, ChatHistorySchema, GroupMemory
from .utils import (
    generate_file_hash,
    check_and_compress_image_bytes,
    process_and_vectorize_session_chats,
)
from .config import Config
from .memory import DB

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

_DEFAULT_DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class PermanentMultimodalError(Exception):
    """Provider rejected the media request and retrying will not help."""


def _pick_api_key(*values: str) -> str:
    for value in values:
        if value and value.strip():
            return value.strip()
    return ""


_summary_api_key = _pick_api_key(plugin_config.summary_api_key, plugin_config.qwen_token, plugin_config.openai_token)
summary_model = (
    ChatOpenAI(
        model=plugin_config.summary_model,
        api_key=SecretStr(_summary_api_key),
        base_url=plugin_config.summary_base_url or _DEFAULT_DASHSCOPE_BASE_URL,
        temperature=0.3,
        max_completion_tokens=800,
    )
    if plugin_config.summary_model and _summary_api_key
    else None
)

_multimodal_api_key = _pick_api_key(plugin_config.multimodal_api_key, plugin_config.qwen_token)
multimodal_model = (
    ChatOpenAI(
        model=plugin_config.multimodal_model,
        api_key=SecretStr(_multimodal_api_key),
        base_url=plugin_config.multimodal_base_url or _DEFAULT_DASHSCOPE_BASE_URL,
        temperature=0.01,
    )
    if plugin_config.multimodal_model and _multimodal_api_key
    else None
)


def _extract_model_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                text = item.strip()
                if text:
                    parts.append(text)
                continue
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
        return "\n".join(parts).strip()
    return ""


def _strip_code_fence(text: str) -> str:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _extract_json_object(text: str) -> dict[str, Any] | None:
    cleaned = _strip_code_fence(text)
    if not cleaned:
        return None

    try:
        payload = json.loads(cleaned)
        return payload if isinstance(payload, dict) else None
    except Exception:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or start >= end:
        return None

    try:
        payload = json.loads(cleaned[start : end + 1])
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _image_to_data_uri(file_path: Path) -> str:
    mime_type = mimetypes.guess_type(str(file_path))[0] or "image/png"
    if not mime_type.startswith("image/"):
        mime_type = "image/png"
    payload = base64.b64encode(file_path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{payload}"


def _is_permanent_multimodal_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "error code: 400",
            "status code: 400",
            "http 400",
            "400 bad request",
            "status=400",
        )
    )


async def _call_multimodal_model(prompt: str, file_path: Path, timeout_s: float = 90.0) -> str | None:
    if multimodal_model is None:
        return None

    try:
        response = await asyncio.wait_for(
            multimodal_model.ainvoke(
                [
                    LCSystemMessage(content="你是一个图片分析助手，请严格按照要求输出结果。"),
                    LCHumanMessage(
                        content=[
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": _image_to_data_uri(file_path)}},
                        ]
                    ),
                ]
            ),
            timeout=timeout_s,
        )
        text = _extract_model_text(response.content)
        return text or None
    except asyncio.TimeoutError:
        logger.warning(f"多模态模型调用超时: {file_path}")
        return None
    except Exception as e:
        if _is_permanent_multimodal_error(e):
            logger.warning(f"多模态模型返回不可重试错误 {file_path}: {e}")
            raise PermanentMultimodalError(str(e)) from e
        logger.error(f"多模态模型调用失败 {file_path}: {e}")
        return None


async def _describe_image_short(file_path: Path) -> str:
    prompt = (
        "请用一句中文简短描述这张图片，保留主体和情绪信息，"
        "不要编造细节，不要超过30个字，只输出描述本身。"
    )
    try:
        result = await _call_multimodal_model(prompt, file_path, timeout_s=60.0)
    except PermanentMultimodalError:
        return "[图片]"
    if not result:
        return "[图片]"
    result = _strip_code_fence(result).replace("\n", " ").strip()
    if not result:
        return "[图片]"
    return result[:60]


async def _analyze_meme_image(file_path: Path) -> tuple[bool | None, str]:
    prompt = """
你是一个专业的表情包分析员。请分析这张图片，并严格输出 JSON。

判断规则：
1. 如果是带梗、表情、配字吐槽、二创、熊猫头、二次元 reaction image，is_meme=true。
2. 如果只是普通照片、聊天截图、风景、证件照、商品图、长篇文档截图，is_meme=false。
3. description 需要概括主体、文字内容和表达情绪，方便后续搜索。

只输出下面格式，不要输出其他内容：
{
  "is_meme": true,
  "description": "熊猫头流泪，配文我太难了，表达委屈和无奈"
}
"""
    result = await _call_multimodal_model(prompt, file_path, timeout_s=90.0)
    payload = _extract_json_object(result or "")
    if payload is None:
        logger.warning(f"多模态模型未返回合法 JSON: {file_path}")
        return None, ""

    is_meme = payload.get("is_meme")
    description = str(payload.get("description", "") or "").strip()
    if not isinstance(is_meme, bool):
        return None, description
    return is_meme, description


async def _call_summary_model(existing_summary: str, chat_text: str) -> str | None:
    if summary_model is None:
        return None

    system_prompt = """你是一个群文化分析师。你的任务是维护一份关于QQ群的认知档案。
档案包含：群内常见话题、活跃成员特征、内部梗/黑话、群文化氛围。
规则：
1. 只能基于提供的聊天记录总结，不要凭空发明内容
2. 保留档案中仍然有效的内容，用新聊天补充或修正旧内容
3. 如果某个内容长期没有聊天印证，可删除
4. 输出完整更新后的档案，不超过500字，不要输出任何其他内容"""
    history_intro = "（无，这是首次建档）" if not existing_summary.strip() else existing_summary
    user_prompt = f"【现有档案】\n{history_intro}\n\n【最新聊天记录】\n{chat_text}\n\n请输出更新后的档案："

    try:
        response = await asyncio.wait_for(
            summary_model.ainvoke(
                [
                    LCSystemMessage(content=system_prompt),
                    LCHumanMessage(content=user_prompt),
                ]
            ),
            timeout=120.0,
        )
        text = _extract_model_text(response.content)
        return text or None
    except asyncio.TimeoutError:
        logger.warning("群体认知档案更新超时，跳过本轮")
        return None
    except Exception as e:
        logger.error(f"群体认知档案更新失败: {e}")
        return None

switch_file = plugin_data_dir / "switch.json"
_enabled = True
if switch_file.exists():
    try:
        _enabled = json.loads(switch_file.read_text(encoding="utf-8")).get("enabled", True)
    except Exception:
        _enabled = True

def _is_enabled() -> bool:
    return _enabled


def _set_enabled(value: bool) -> None:
    global _enabled
    _enabled = value
    switch_file.write_text(json.dumps({"enabled": value}, ensure_ascii=False), encoding="utf-8")


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


def _extract_reply_message_id_from_event(event: Event) -> str | None:
    def _pick_reply_id(obj: Any) -> str | None:
        if obj is None:
            return None
        if isinstance(obj, dict):
            rid = obj.get("id") or obj.get("message_id")
            if rid is not None and str(rid).strip():
                return str(rid).strip()
        rid = getattr(obj, "id", None) or getattr(obj, "message_id", None)
        if rid is not None and str(rid).strip():
            return str(rid).strip()
        return None

    try:
        # Some adapters may expose a structured reply object directly.
        rid = _pick_reply_id(getattr(event, "reply", None))
        if rid:
            return rid

        raw_msg = getattr(event, "message", None) or getattr(event, "original_message", None)
        if raw_msg:
            for seg in raw_msg:
                seg_type = getattr(seg, "type", None)
                if seg_type != "reply":
                    continue
                rid = _pick_reply_id(getattr(seg, "data", None)) or _pick_reply_id(seg)
                if rid:
                    return rid

            # OneBot v11 常见字符串形态：[reply:id=123456]
            text = str(raw_msg)
            m = re.search(r"(?:\[)?reply:id=(\d+)(?:\])?", text)
            if m:
                return m.group(1)

        # Last fallback: parse serialized event text.
        event_text = str(event)
        m = re.search(r"(?:\[)?reply:id=(\d+)(?:\])?", event_text)
        if m:
            return m.group(1)
    except Exception:
        return None
    return None


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


_MEME_BLOCK_FEEDBACK_KEYWORDS = (
    "不可以",
    "不行",
    "不能",
    "别发",
    "别再发",
    "不要这张",
)

_MEME_BLOCK_CONFIRM_TEMPLATES = (
    "{bot_name}知道错了...达咩!",
    "{bot_name}不会再发这个表情包了...",
    "果面呐噻,{bot_name}发错表情包了...",
    "{bot_name}有说什么奇怪的话吗？",
)


def _user_id_aliases(raw_id: str | None) -> set[str]:
    raw = str(raw_id or "").strip()
    if not raw:
        return set()

    aliases: set[str] = {raw}
    if ":" in raw:
        aliases.add(raw.rsplit(":", 1)[-1].strip())

    # 兜底抽取末尾数字，兼容 onebot/qq 前缀差异
    m = re.search(r"(\d+)$", raw)
    if m:
        aliases.add(m.group(1))
    return {a for a in aliases if a}


def _is_superuser_id(user_id: str | None) -> bool:
    uid_aliases = _user_id_aliases(user_id)
    if not uid_aliases:
        return False

    try:
        superusers = {str(i).strip() for i in get_driver().config.superusers}
    except Exception:
        superusers = set()

    for su in superusers:
        if uid_aliases & _user_id_aliases(su):
            return True
    return False


def _is_meme_block_feedback(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    return any(k in t for k in _MEME_BLOCK_FEEDBACK_KEYWORDS)


async def _try_auto_block_replied_meme(
    db_session: async_scoped_session,
    session_id: str,
    sender_user_id: str | None,
    event: Event,
    bot: Bot,
    reply_message_id: str | None,
    plain_text: str,
) -> bool:
    if not _is_meme_block_feedback(plain_text):
        return False

    if not reply_message_id:
        logger.info(f"[MemeBlock] feedback_hit_but_reply_id_missing session_id={session_id} sender={sender_user_id}")
        return False

    try:
        is_superuser = await SUPERUSER(bot, event)
    except Exception:
        is_superuser = _is_superuser_id(sender_user_id)

    if not is_superuser:
        logger.info(
            f"[MemeBlock] feedback_hit_but_not_superuser session_id={session_id} "
            f"sender={sender_user_id} reply_to={reply_message_id}"
        )
        return False

    stmt = (
        Select(ChatHistory)
        .where(
            ChatHistory.session_id == session_id,
            ChatHistory.content_type == "bot",
            ChatHistory.content.like(f"%id:{reply_message_id}%"),
        )
        .order_by(desc(ChatHistory.created_at))
        .limit(1)
    )
    target = (await db_session.execute(stmt)).scalar_one_or_none()

    if target is None:
        stmt = (
            Select(ChatHistory)
            .where(
                ChatHistory.session_id == session_id,
                ChatHistory.content_type == "bot",
                ChatHistory.content.like(f"%id: {reply_message_id}%"),
            )
            .order_by(desc(ChatHistory.created_at))
            .limit(1)
        )
        target = (await db_session.execute(stmt)).scalar_one_or_none()

    if not target:
        logger.info(
            f"[MemeBlock] feedback_hit_but_target_not_found "
            f"session_id={session_id} reply_to={reply_message_id}"
        )
        return False

    media_id: int | None = int(target.media_id) if target.media_id is not None else None
    if media_id is None:
        marker = "图片描述是:"
        sent_content = target.content or ""
        if marker in sent_content:
            desc_text = sent_content.split(marker, 1)[1].strip()
            if desc_text:
                by_desc = (
                    Select(MediaStorage)
                    .where(MediaStorage.description == desc_text)
                    .order_by(desc(MediaStorage.media_id))
                    .limit(1)
                )
                media_by_desc = (await db_session.execute(by_desc)).scalar_one_or_none()
                if media_by_desc:
                    media_id = int(media_by_desc.media_id)

    if media_id is None:
        logger.info(
            f"[MemeBlock] feedback_hit_but_media_id_missing "
            f"session_id={session_id} reply_to={reply_message_id}"
        )
        return False

    media = (
        await db_session.execute(Select(MediaStorage).where(MediaStorage.media_id == media_id))
    ).scalar_one_or_none()
    if not media:
        return False

    media_id_int = int(media_id)
    changed = not bool(media.blocked)
    if changed:
        media.blocked = True
        db_session.add(media)
        try:
            await db_session.commit()
        except Exception as e:
            await db_session.rollback()
            logger.error(
                f"[MemeBlock] update blocked flag failed: media_id={media_id_int} "
                f"err={type(e).__name__}: {e}"
            )
            return False
        if DB.enabled:
            try:
                await DB.delete_media(media_id_int)
            except Exception as e:
                logger.warning(f"[MemeBlock] 从 Qdrant 删除拉黑表情包失败 media_id={media_id_int}: {e}")
    logger.info(
        f"[MemeBlock] superuser={sender_user_id} session_id={session_id} "
        f"reply_to={reply_message_id} media_id={media_id_int} changed={changed}"
    )
    try:
        bot_name = (plugin_config.bot_name or "bot").strip() or "bot"
        if changed:
            tip = random.choice(_MEME_BLOCK_CONFIRM_TEMPLATES).format(bot_name=bot_name)
        else:
            tip = f"该表情包已在黑名单中（id={media_id_int}）。"
        await UniMessage.text(tip).send(reply_to=True)
    except Exception as e:
        logger.warning(f"[MemeBlock] 发送拉黑提示失败: {type(e).__name__}: {e}")
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
        if not DB.enabled:
            qdrant_line = "disabled"
        else:
            ok, detail = await DB.healthcheck()
            qdrant_line = detail if ok else f"down ({detail})"

        await ai.finish(f"ai status: {status}\nqdrant: {qdrant_line}")
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
    reply_to_message_id: str | None = None
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
            rid = str(getattr(i, "id", "") or "").strip()
            if not rid:
                data = getattr(i, "data", None)
                if isinstance(data, dict):
                    rid = str(data.get("id") or data.get("message_id") or "").strip()
            if not rid:
                rid = str(getattr(i, "message_id", "") or "").strip()
            if rid and not reply_to_message_id:
                reply_to_message_id = rid
                content += "回复id:" + rid
        if i.type == "text":
            content += i.text
            is_text = True

    if not reply_to_message_id:
        reply_to_message_id = _extract_reply_message_id_from_event(event)
    if not reply_to_message_id:
        m = re.search(r"(?:\[)?reply:id=(\d+)(?:\])?", str(msg))
        if m:
            reply_to_message_id = m.group(1)

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

    # ========== 步骤2: 处理图片消息（后台异步，不阻塞回复决策） ==========
    image_content_prefix = f"id: {get_message_id()}\n"
    for img in imgs:
        asyncio.create_task(
            _process_image_task(img, event, bot, state, session, user_name, image_content_prefix)
        )

    plain_text = (msg.extract_plain_text() or event.get_plaintext() or "").strip()
    if await _try_auto_block_replied_meme(
        db_session,
        session.scene.id,
        session.user.id,
        event,
        bot,
        reply_to_message_id,
        plain_text,
    ):
        return

    if not _is_enabled():
        return

    # ========== 步骤3: 决定是否回复 ==========
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
        await handle_reply_logic(
            db_session,
            session,
            plugin_config.bot_name,
            user_id,
            user_name,
            to_me,
        )

    await db_session.commit()


async def _process_image_task(
    img: Image,
    event: Event,
    bot: Bot,
    state: T_State,
    session: Uninfo,
    user_name: str | None,
    content_prefix: str,
) -> None:
    try:
        async with get_session() as db_session:
            await process_image_message(db_session, img, event, bot, state, session, user_name, content_prefix)
    except Exception as e:
        logger.error(f"后台处理图片失败: {e}")


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

    image_format = img.id.split(".")[-1] if "." in img.id else "jpg"
    if not image_format:
        image_format = "jpg"

    try:
        pic = await asyncio.wait_for(image_fetch(event, bot, state, img), timeout=15.0)
    except asyncio.TimeoutError:
        logger.warning("下载图片超时，跳过")
        return

    try:
        pic = await asyncio.to_thread(check_and_compress_image_bytes, pic, image_format=image_format.upper())
        file_hash = generate_file_hash(pic)
        file_name = f"{file_hash}.{image_format}"
        file_path = pic_dir / file_name

        if not file_path.exists():
            file_path.write_bytes(pic)

        stmt = Select(MediaStorage).where(MediaStorage.file_hash == file_hash)
        media_obj = (await db_session.execute(stmt)).scalar_one_or_none()
        if media_obj:
            media_obj.references += 1
            db_session.add(media_obj)
        else:
            try:
                async with db_session.begin_nested():
                    media_obj = MediaStorage(
                        file_hash=file_hash,
                        file_path=file_name,
                        references=1,
                        description="[图片]",
                    )
                    db_session.add(media_obj)
                    await db_session.flush()
            except Exception:
                media_obj = (await db_session.execute(stmt)).scalar_one_or_none()
                if media_obj is None:
                    raise
                logger.info(f"图片并发插入冲突 {file_hash}，转为更新模式")
                media_obj.references += 1
                db_session.add(media_obj)

        await db_session.flush()

        image_description = "[图片]"
        if media_obj:
            cached_description = (media_obj.description or "").strip()
            if cached_description and cached_description != "[图片]":
                image_description = cached_description
            else:
                image_description = await _describe_image_short(file_path)
                media_obj.description = image_description
                db_session.add(media_obj)

            chat_history = ChatHistory(
                session_id=session.scene.id,
                user_id=session.user.id,
                content_type=content_type,
                content=content + image_description,
                user_name=user_name or "",
                media_id=media_obj.media_id,
            )
            db_session.add(chat_history)

        await db_session.commit()

    except Exception as e:
        logger.error(f"处理图片失败: {e}")
        await db_session.rollback()


def _clean_gate_text(content: str) -> str:
    lines = []
    for raw_line in (content or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("id:") or line.startswith("回复id:"):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


async def handle_reply_logic(
    db_session,
    session: Uninfo,
    bot_name: str,
    user_id: str,
    user_name: str | None,
    is_tome: bool,
):
    """处理回复逻辑"""
    try:
        # 非 @bot / 非明确点名时，先做一次轻量前置判断，避免频繁启动主 Agent。
        recent_msgs = (
            await db_session.execute(
                Select(ChatHistory)
                .where(ChatHistory.session_id == session.scene.id)
                .order_by(ChatHistory.msg_id.desc())
                .limit(3)
            )
        ).scalars().all()
        recent_msgs = recent_msgs[::-1]

        if not recent_msgs:
            return

        history_summary = ""
        for msg in recent_msgs:
            if msg.content_type == "image":
                history_summary += f"{msg.user_name}: [发送了一张图片]\n"
            else:
                history_summary += f"{msg.user_name}: {_clean_gate_text(msg.content)}\n"

        current_msg_text = (
            _clean_gate_text(recent_msgs[-1].content)
            if recent_msgs[-1].content_type == "text"
            else "[图片]"
        )

        if not is_tome:
            should_reply = await check_if_should_reply(
                history_summary,
                current_msg_text,
                bot_name,
            )
            if not should_reply:
                logger.debug(f"前置判断拒绝回复 session={session.scene.id}")
                return

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
            await asyncio.wait_for(
                choice_response_strategy(
                    db_session,
                    session.scene.id,
                    last_msg,
                    user_id,
                    user_name,
                    plugin_config.personality_setting,
                ),
                timeout=120.0,
            )
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
    if not _is_enabled() or not DB.enabled:
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


@scheduler.scheduled_job("interval", minutes=30, max_instances=1, coalesce=True, id="vectorize_media")
async def vectorize_media():
    if not _is_enabled() or not DB.enabled:
        return
    if multimodal_model is None:
        logger.warning("未配置 multimodal_model，跳过媒体向量化")
        return

    async with get_session() as db_session:
        medias_res = await db_session.execute(
            Select(MediaStorage).where(
                MediaStorage.references >= 3,
                MediaStorage.vectorized.is_(False),
                MediaStorage.blocked.is_(False),
            )
        )
        media_ids = [media.media_id for media in medias_res.scalars().all()]
        logger.info(f"待向量化媒体数量: {len(media_ids)}")

        for media_id in media_ids:
            media = await db_session.get(MediaStorage, media_id)
            if media is None:
                continue
            current_media_id = int(media_id)
            current_file_name = media.file_path
            try:
                file_path = pic_dir / current_file_name
                if not file_path.exists():
                    logger.warning(f"文件不存在: {file_path}")
                    media.vectorized = True
                    db_session.add(media)
                    await db_session.commit()
                    continue

                try:
                    is_meme, description = await _analyze_meme_image(file_path)
                except PermanentMultimodalError as e:
                    logger.warning(f"图片分析返回不可重试错误，跳过 media_id={current_media_id}: {e}")
                    media.vectorized = True
                    db_session.add(media)
                    await db_session.commit()
                    continue
                if is_meme is None:
                    logger.warning(f"图片分析失败，保留待重试状态 media_id={current_media_id}")
                    continue

                if not is_meme:
                    media.vectorized = True
                    db_session.add(media)
                    await db_session.commit()
                    logger.info(f"图片 {current_media_id} 被判定为非表情包，跳过向量化")
                    continue

                if description:
                    media.description = description
                media_description = media.description or description or "[图片]"

                await DB.insert_media(
                    current_media_id,
                    [str(file_path)],
                    description=media_description,
                    file_path=str(file_path),
                    blocked=bool(media.blocked),
                )
                media.vectorized = True
                db_session.add(media)
                await db_session.commit()
                logger.info(f"媒体向量化成功 media_id={current_media_id}")
            except Exception as e:
                logger.error(f"处理媒体 {current_media_id} 失败: {e}")
                await db_session.rollback()
                continue

        logger.info("向量化媒体完成")


@scheduler.scheduled_job("interval", minutes=35, max_instances=1, coalesce=True, id="clear_cache")
async def clear_cache_pic():
    async with get_session() as db_session:
        result = await db_session.execute(
            Select(MediaStorage).where(
                MediaStorage.references < 3,
                MediaStorage.created_at < datetime.datetime.now() - datetime.timedelta(days=15),
            )
        )
        medias = result.scalars().all()

        records_to_delete = []
        for media in medias:
            current_media_id = int(media.media_id)
            current_file_name = media.file_path
            try:
                file_path = Path(pic_dir / current_file_name)
                # use pathlib unlink with missing_ok=True to avoid raising if missing
                await asyncio.to_thread(file_path.unlink, True)
                records_to_delete.append((media, current_media_id, current_file_name))
                logger.debug(f"删除文件: {file_path}")
            except Exception as e:
                logger.error(f"删除文件失败 {current_file_name}: {e}")
                records_to_delete.append((media, current_media_id, current_file_name))

        for media, current_media_id, current_file_name in records_to_delete:
            try:
                if DB.enabled:
                    try:
                        await DB.delete_media(current_media_id)
                    except Exception as e:
                        logger.warning(f"删除 Qdrant 媒体向量失败 media_id={current_media_id}: {e}")
                await db_session.delete(media)
            except Exception as e:
                logger.error(f"删除数据库记录失败 media_id={current_media_id}, file_path={current_file_name}: {e}")

        await db_session.commit()
        if records_to_delete:
            logger.info(f"成功清理 {len(records_to_delete)} 个媒体记录")
        else:
            logger.info("没有需要清理的媒体文件")

        known_files_result = await db_session.execute(Select(MediaStorage.file_path))
        known_files = {row[0] for row in known_files_result.all()}

        disk_files = await asyncio.to_thread(lambda: list(pic_dir.iterdir()))
        orphaned = [file_path for file_path in disk_files if file_path.is_file() and file_path.name not in known_files]

        for file_path in orphaned:
            try:
                await asyncio.to_thread(file_path.unlink, True)
                logger.debug(f"删除孤立文件: {file_path.name}")
            except Exception as e:
                logger.error(f"删除孤立文件失败 {file_path.name}: {e}")

        if orphaned:
            logger.info(f"成功清理 {len(orphaned)} 个孤立文件")


async def _update_single_group_memory(db_session: async_scoped_session, session_id: str) -> None:
    stmt = Select(GroupMemory).where(GroupMemory.session_id == session_id)
    record = (await db_session.execute(stmt)).scalar_one_or_none()

    total_count = (
        await db_session.execute(
            Select(sqlfunc.count(ChatHistory.msg_id)).where(ChatHistory.session_id == session_id)
        )
    ).scalar_one()
    last_count = record.msg_count_at_last_update if record else 0
    new_msg_count = total_count - last_count

    if record and new_msg_count < 100:
        time_since = datetime.datetime.now() - record.updated_at
        if time_since.total_seconds() < 6 * 3600:
            logger.info(f"群 {session_id} 无需更新档案（新增 {new_msg_count} 条）")
            return

    cutoff = record.updated_at if record else datetime.datetime.min
    recent_msgs = (
        await db_session.execute(
            Select(ChatHistory)
            .where(
                ChatHistory.session_id == session_id,
                ChatHistory.created_at > cutoff,
                ChatHistory.content_type.in_(["text", "bot", "image"]),
            )
            .order_by(ChatHistory.created_at)
            .limit(200)
        )
    ).scalars().all()
    if not recent_msgs:
        return

    chat_text = "\n".join(
        f"[{msg.created_at.strftime('%m-%d %H:%M')}] {msg.user_name}: {msg.content[:120]}"
        for msg in recent_msgs
    )

    existing_summary = record.summary if record else ""
    new_summary = await _call_summary_model(existing_summary, chat_text)
    if not new_summary:
        return

    if not record:
        record = GroupMemory(
            session_id=session_id,
            summary=new_summary,
            msg_count_at_last_update=total_count,
        )
        db_session.add(record)
    else:
        record.summary = new_summary
        record.msg_count_at_last_update = total_count

    await db_session.commit()
    logger.info(f"群体认知档案更新成功 session_id={session_id}")


@scheduler.scheduled_job("interval", hours=6, max_instances=1, coalesce=True, id="update_group_memory")
async def update_group_memory():
    if not _is_enabled() or summary_model is None:
        return

    async with get_session() as db_session:
        time_threshold = datetime.datetime.now() - datetime.timedelta(days=1)
        stmt = Select(ChatHistory.session_id.distinct()).where(ChatHistory.created_at > time_threshold)
        session_ids = (await db_session.execute(stmt)).scalars().all()

    if not session_ids:
        return

    sem = asyncio.Semaphore(5)

    async def _update_one(session_id: str) -> None:
        async with sem:
            async with get_session() as db_session:
                try:
                    await _update_single_group_memory(db_session, session_id)
                except Exception as e:
                    logger.error(f"更新群体认知档案失败 session_id={session_id}: {e}")

    await asyncio.gather(*[_update_one(session_id) for session_id in session_ids])
