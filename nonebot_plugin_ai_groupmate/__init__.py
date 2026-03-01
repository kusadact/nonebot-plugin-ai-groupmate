import random
import asyncio
import datetime
import traceback
import json
import re
from io import BytesIO
from pathlib import Path
from typing import Any

import jieba
from nonebot import logger, require, on_command, on_message, get_plugin_config, get_driver
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
from sqlalchemy import Select, desc
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

    # ========== 步骤2: 处理图片消息（耗时） ==========
    for img in imgs:
        await process_image_message(db_session, img, event, bot, state, session, user_name, f"id: {get_message_id()}\n")

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
