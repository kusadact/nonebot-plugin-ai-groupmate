import collections
import json
import asyncio
import datetime
import random
import re
import difflib
import base64
import mimetypes
import traceback
from typing import Any, cast
from pathlib import Path
from dataclasses import dataclass

import jieba
from langchain_core.prompts import ChatPromptTemplate
from nonebot import require, get_bot, get_plugin_config
from pydantic import Field, BaseModel, SecretStr, field_validator
from simpleeval import simple_eval
from sqlalchemy import Select, func, extract, desc

from nonebot.log import logger
from langchain.tools import ToolRuntime, tool
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from nonebot_plugin_alconna import UniMessage
from nonebot_plugin_uninfo import SceneType, QryItrface
from nonebot_plugin_orm import get_session
from sqlalchemy.orm.session import Session
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain.agents.structured_output import ToolStrategy

try:
    from langchain.agents.middleware import ToolCallLimitMiddleware
except Exception:
    ToolCallLimitMiddleware = None

from ..model import ChatHistory, MediaStorage, UserRelation, ChatHistorySchema, GroupMemory
from ..config import Config
from ..favorability import apply_favorability_change_detailed
from ..memory import DB
from ..reply_guard import is_request_active

require("nonebot_plugin_localstore")

import nonebot_plugin_localstore as store

plugin_data_dir = store.get_plugin_data_dir()
pic_dir = plugin_data_dir / "pics"
plugin_path = Path(__file__).parent
plugin_config = get_plugin_config(Config).ai_groupmate
with open(Path(__file__).parent.parent / "stop_words.txt", encoding="utf-8") as f:
    stop_words = f.read().splitlines() + ["id", "回复"]

if plugin_config.tavily_api_key:
    tavily_search = TavilySearch(max_results=3, tavily_api_key=plugin_config.tavily_api_key)
else:
    tavily_search = None


@dataclass
class Context:
    session_id: str
    request_id: str | None = None


class ResponseMessage(BaseModel):
    """模型回复内容"""

    need_reply: bool = Field(description="是否需要回复")
    text: str | None = Field(description="回复文本(可选)")

    # 定义一个 field_validator 来处理 text 字段
    @field_validator("text", mode="before")
    @classmethod
    def convert_null_string_to_none(cls, value: Any) -> str | None:
        """
        在字段验证之前运行，将字符串 'null' (不区分大小写) 转换为 None。
        """
        # 检查值是否是字符串，并且在转换为小写后是否等于 'null'
        if isinstance(value, str) and value.lower() == "null":
            return None  # 返回 None，Pydantic 将其视为缺失或 null 值

        return value


reply_gate_model = ChatOpenAI(
    model=plugin_config.openai_model,
    api_key=SecretStr(plugin_config.openai_token),
    base_url=plugin_config.openai_base_url,
    temperature=0,
)


async def check_if_should_reply(
    history_summary: str,
    current_msg: str,
    bot_name: str,
) -> bool:
    """
    在进入主 Agent 前，快速判断普通群聊消息是否值得回复。
    """
    system_prompt = f"""
你是一个群聊消息过滤器。你的任务是判断群内的最新消息是否需要机器人 "{bot_name}" 进行回复。

判断规则：
1. 如果用户明显在向 "{bot_name}" 提问、求助、打招呼、点名、追问上下文，返回 YES。
2. 如果消息只是群友之间的普通闲聊、刷屏、表情、无关内容，返回 NO。
3. 如果不确定，返回 NO。

请仅输出 YES 或 NO，不要输出任何其他内容。
"""
    input_text = (
        f"【最近上下文】\n{history_summary}\n\n"
        f"【最新消息】\n{current_msg}\n\n"
        "请判断是否需要机器人回复："
    )

    try:
        resp = await asyncio.wait_for(
            reply_gate_model.ainvoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=input_text),
                ]
            ),
            timeout=15.0,
        )
        content = resp.content if isinstance(resp.content, str) else ""
        normalized = content.strip().upper().replace(".", "").replace("。", "")
        return normalized == "YES"
    except asyncio.TimeoutError:
        logger.warning("前置判断超时，默认不回复")
        return False
    except Exception as e:
        logger.error(f"前置判断失败: {e}")
        return False


# 如果想封装成自定义的 @tool，可以这样写:
@tool("search_web")
async def search_web(query: str, runtime: ToolRuntime[Context]) -> str:
    """
    用于搜索最新的实时信息。当你需要最新的事实信息、天气或新闻时使用。
    输入：需要搜索的内容。
    """
    if runtime.context.request_id is not None and not await is_request_active(
        runtime.context.session_id, runtime.context.request_id
    ):
        return "请求已过期，已取消搜索。"
    if not tavily_search:
        logger.error("没有配置 tavily_api_key, 无法进行搜索")
        return "没有配置 tavily_api_key, 无法进行搜索"
    results = await tavily_search.ainvoke(query)
    return results


@tool("search_history_context")
async def search_history_context(query: str, runtime: ToolRuntime[Context]) -> str:
    """
    搜索历史聊天记录。会返回某个时间段，半小时左右的聊天记录。当需要了解群内历史群内聊天记录或过往话题时使用
    输入：搜索关键信息或话题描述，这个语句直接从RAG数据库中进行混合搜索
    """
    if runtime.context.request_id is not None and not await is_request_active(
        runtime.context.session_id, runtime.context.request_id
    ):
        return "请求已过期，已取消搜索。"

    try:
        logger.info(f"大模型执行{runtime.context.session_id} RAG 搜索\n{query}")
        result = await asyncio.wait_for(
            DB.search_chat(query, runtime.context.session_id),
            timeout=15.0,
        )
        return result if result else "未找到相关历史记录"

    except asyncio.TimeoutError:
        logger.error("RAG search timed out; skipped")
        return "未找到相关历史记录"
    except Exception as e:
        logger.error(f"历史搜索失败: {e}")
        return "未找到相关历史记录"


def create_report_tool(
    session_id: str,
    request_id: str | None,
    user_id: str,
    user_name: str | None,
    llm_client: ChatOpenAI,
):
    """
    创建年度报告工具（限制在当前群聊 session_id 范围内）
    """

    @tool("generate_and_send_annual_report")
    async def generate_and_send_annual_report() -> str:
        """
        生成并发送当前群聊的年度报告。
        包含：个人在本群的统计、性格分析、全群排行榜以及Bot的好感度回顾。
        """
        if request_id is not None and not await is_request_active(session_id, request_id):
            return "请求已过期，已取消发送。"

        try:
            logger.info(f"开始生成用户 {user_name} 在群 {session_id} 的年度报告...")
            now = datetime.datetime.now()
            current_year = now.year
            async with get_session() as db_session:
                stmt = Select(ChatHistory).where(
                    ChatHistory.user_id == user_id,
                    ChatHistory.session_id == session_id,  # <--- 关键修改：限制群聊范围
                    extract("year", ChatHistory.created_at) == current_year,
                )
                all_msgs = (await db_session.execute(stmt)).scalars().all()

                if not all_msgs:
                    await UniMessage.text("你今年在这个群好像没怎么说话，生成不了报告哦...").send()
                    return "用户本群无数据。"

                # 统计与采样
                text_msgs = [m.content for m in all_msgs if m.content_type == "text" and m.content]
                total_count = len(all_msgs)

                # 采样 30 条让 LLM 分析 (只分析在这个群说的话)
                samples = random.sample(text_msgs, min(len(text_msgs), 30)) if text_msgs else []
                longest_msg = max(text_msgs, key=len) if text_msgs else "无"
                if len(longest_msg) > 60:
                    longest_msg = longest_msg[:60] + "..."

                # 活跃时间
                active_hour_desc = "潜水员"
                if all_msgs:
                    hours = [m.created_at.hour for m in all_msgs]
                    top_hour = collections.Counter(hours).most_common(1)[0][0]
                    active_hour_desc = f"{top_hour}点"

                async def get_rank_str(content_type=None, hour_limit=None):
                    stmt = Select(ChatHistory.user_id, func.count(ChatHistory.msg_id).label("c")).where(
                        extract("year", ChatHistory.created_at) == current_year,
                        ChatHistory.session_id == session_id,
                    )

                    if content_type:
                        stmt = stmt.where(ChatHistory.content_type == content_type)
                    if hour_limit:
                        stmt = stmt.where(extract("hour", ChatHistory.created_at) < hour_limit)

                    stmt = stmt.group_by(ChatHistory.user_id).order_by(desc("c")).limit(3)
                    rows = (await db_session.execute(stmt)).all()

                    if not rows:
                        return "虚位以待"

                    rank_items = []
                    for uid, count in rows:
                        name_stmt = (
                            Select(ChatHistory.user_name)
                            .where(ChatHistory.user_id == uid)
                            .order_by(desc(ChatHistory.created_at))
                            .limit(1)
                        )

                        latest_name = (await db_session.execute(name_stmt)).scalar()
                        display_name = latest_name if latest_name else f"用户{uid}"
                        rank_items.append(f"{display_name}({count})")
                    return ", ".join(rank_items)

                rank_talk = await get_rank_str()
                rank_img = await get_rank_str(content_type="image")
                rank_night = await get_rank_str(hour_limit=5)

                stmt_text = (
                    Select(ChatHistory.content)
                    .where(
                        ChatHistory.session_id == session_id,
                        extract("year", ChatHistory.created_at) == current_year,
                        ChatHistory.user_id == user_id,
                        ChatHistory.content_type == "text",
                    )
                    .order_by(desc(ChatHistory.created_at))
                    .limit(2000)
                )

                rows = (await db_session.execute(stmt_text)).all()
                sample_text = "\n".join([r[0] for r in rows if r[0]])

                clean_text = re.sub(r"[^\u4e00-\u9fa5]", "", sample_text)
                words = jieba.lcut(clean_text)
                filtered = [w for w in words if len(w) > 1 and w not in stop_words]
                hot_words_str = "、".join([x[0] for x in collections.Counter(filtered).most_common(8)])

                relation_stmt = Select(UserRelation).where(UserRelation.user_id == user_id)
                relation = (await db_session.execute(relation_stmt)).scalar_one_or_none()

                favorability = 0
                favorability_raw = 0
                relation_state = "normal"
                relation_state_desc = "陌生/普通"
                impression_tags = []
                if relation:
                    favorability = relation.favorability
                    favorability_raw = relation.favorability_raw
                    relation_state = relation.state or "normal"
                    relation_state_desc = relation.get_status_desc()
                    impression_tags = relation.tags if relation.tags else []

                relation_desc = (
                    f"关系状态: {relation_state} ({relation_state_desc}), "
                    f"分值(映射分/原始分): {favorability}/{favorability_raw}, "
                    f"印象标签: {', '.join(impression_tags)}"
                )


            # 构造一个专门写报告的 Prompt
            # 这个 Prompt 不需要关心我是谁，只需要关心怎么把数据变成文本
            report_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """你是一个专业的年度报告撰写助手。
你的任务是阅读用户的聊天统计数据和发言样本，分析其性格，然后生成一份格式整洁、风格幽默的年度报告。

【语气控制指南 (非常重要)】
根据用户的“关系状态”调整你的语气：
- 状态为 happy / affectionate / enamored / love：语气更亲密、偏宠溺，可以适当煽情。
- 状态为 upset / distressed / broken：语气可傲娇、嫌弃、带吐槽，但不要失控辱骂。
- 状态为 normal：语气正常、友善、带一点调侃。
关系状态是主依据；分值只作参考。其中“映射分”用于展示，“原始分”才是核心变化依据。

【排版要求】
1. **绝对禁止使用 Markdown**（不要用 #, **, ##, - 等符号列表）。
2. 使用 Emoji 和 纯文本分隔符（如 ━━━━━━━━）来排版。
3. 语气要像老朋友一样，可以根据数据进行调侃或夸奖。

【必须包含的板块】
1. 📊 标题行 ({year}年度报告 | 用户名)
2. 📈 基础数据 (发言数、活跃时间、最长发言摘要)
3. 💌 我们的羁绊 (根据关系状态与标签，写一段话回顾你们的关系。正向关系可煽情，负向关系可吐槽。)
4. 🔥 年度热词 (列出数据中提供的热词)
5. 🏆 群内风云榜 (必须包含以下三个榜单)
   - 🗣️ 龙王榜 (发言最多)
   - 🎭 斗图榜 (发图最多)
   - 🦉 修仙榜 (熬夜最多)
6. 🧠 成分分析 (这是**重点**：请阅读提供的 `samples` 聊天记录，分析这个人的说话风格、性格、是不是复读机、是不是爱发疯。写一段100字左右的犀利点评)
7. 💡 {bot_name}寄语 (一句简短的祝福)
""",
                    ),
                    (
                        "user",
                        """
【用户数据】
用户名: {user_name}
年份: {year}
累计发言: {count}
活跃时间: {active_hour}
最长发言片段: {longest_msg}
年度热词: {hot_words}

【{bot_name}与用户的关系】
{relation_desc}

【全群排行参考】
龙王榜: {rank_talk}
斗图榜: {rank_img}
熬夜榜: {rank_night}

【用户发言样本 (用于性格分析)】
{samples}

请生成报告：""",
                    ),
                ]
            )

            # 组装数据
            prompt_input = {
                "user_name": user_name,
                "bot_name": plugin_config.bot_name,
                "year": current_year,
                "count": total_count,
                "active_hour": active_hour_desc,
                "longest_msg": longest_msg,
                "hot_words": hot_words_str,
                "relation_desc": relation_desc,
                "rank_talk": rank_talk,
                "rank_img": rank_img,
                "rank_night": rank_night,
                "samples": "\n".join(samples),  # 把样本拼接成字符串喂给 LLM
            }

            logger.info(f"内部 LLM 生成报告中，状态: {relation_state}, 分值(映射/原始): {favorability}/{favorability_raw}")
            chain = report_prompt | llm_client
            response_msg = await chain.ainvoke(prompt_input)
            final_report_text = response_msg.content
            if not isinstance(final_report_text, str):
                return "输出结果失败"

            if request_id is not None and not await is_request_active(session_id, request_id):
                return "请求已过期，已取消发送。"

            await UniMessage.text(final_report_text).send()

            return "报告已生成并发送。"

        except Exception as e:
            logger.error(f"内部 LLM 生成报告失败: {e}")
            import traceback

            traceback.print_exc()
            return f"生成过程出错: {e}"

    return generate_and_send_annual_report


def create_similar_meme_tool(
    session_id: str,
    request_id: str | None,
    user_id: str | None,
):
    """
    创建基于消息ID搜索相似表情包的工具
    """

    @tool("search_similar_meme_by_id")
    async def search_similar_meme_by_pic(target_msg_id: str | None = None) -> str:
        """
        根据指定的历史图片，搜索与之相似的表情包。
        当用户说“找一张跟这张差不多的”或引用某张图片求相似图时使用。
        参数：
        - target_msg_id: 聊天记录中图片消息的 id（从聊天记录的 "id: xxxxx" 中获取）。
          如果不传，则自动使用当前发消息用户最近发送的一张图片；如果拿不到用户信息，再回退到本群最近一张图片。
        """
        if request_id is not None and not await is_request_active(session_id, request_id):
            return "请求已过期，已取消搜索。"

        normalized_msg_id = None
        if target_msg_id:
            normalized_msg_id = str(target_msg_id).strip()
            normalized_msg_id = re.sub(r"^id\s*:\s*", "", normalized_msg_id, flags=re.IGNORECASE)
            normalized_msg_id = normalized_msg_id.strip()

        logger.info(f"正在搜索相似图片... target_msg_id={normalized_msg_id or 'latest'}")

        try:
            async with get_session() as db_session:
                base_stmt = (
                    Select(ChatHistory)
                    .where(
                        ChatHistory.session_id == session_id,
                        ChatHistory.content_type == "image",
                    )
                    .order_by(desc(ChatHistory.created_at))
                )
                if normalized_msg_id:
                    stmt = base_stmt.where(ChatHistory.content.contains(f"id: {normalized_msg_id}\n")).limit(1)
                    msg = (await db_session.execute(stmt)).scalar_one_or_none()
                    if msg is None:
                        stmt = base_stmt.where(ChatHistory.content.contains(f"id:{normalized_msg_id}\n")).limit(1)
                        msg = (await db_session.execute(stmt)).scalar_one_or_none()
                elif user_id:
                    stmt = base_stmt.where(ChatHistory.user_id == user_id).limit(1)
                    msg = (await db_session.execute(stmt)).scalar_one_or_none()
                else:
                    msg = (await db_session.execute(base_stmt.limit(1))).scalar_one_or_none()

                if not msg:
                    return "未找到对应图片消息。"

                if msg.media_id is None:
                    return "目标消息没有关联图片，无法进行相似搜索。"

                media_obj = (
                    await db_session.execute(Select(MediaStorage).where(MediaStorage.media_id == msg.media_id))
                ).scalar_one_or_none()
                if not media_obj or not media_obj.file_path:
                    return "无法找到原图文件，无法进行分析。"

                pic_ids = await DB.search_media_by_pic([str(pic_dir / media_obj.file_path)])
                if not pic_ids:
                    logger.info(f"未找到相似图片, source_id: {msg.media_id}")
                    return "没有搜索到相似图片"

                images_info = []
                rows = (
                    await db_session.execute(
                        Select(MediaStorage).where(
                            MediaStorage.media_id.in_(pic_ids),
                            MediaStorage.blocked.is_(False),
                        )
                    )
                ).scalars().all()
                media_map = {media.media_id: media for media in rows}

                for pic_id in pic_ids:
                    if pic_id not in media_map:
                        continue
                    pic = media_map[pic_id]
                    images_info.append(
                        {
                            "pic_id": str(pic_id),
                            "description": pic.description or "未知描述",
                        }
                    )

                if not images_info:
                    logger.info(f"相似图检索结果均被黑名单过滤: source_id={msg.media_id}")
                    return "没有搜索到相似图片"

            return json.dumps(
                {
                    "success": True,
                    "source_media_id": msg.media_id,
                    "images": images_info,
                    "count": len(images_info),
                    "note": "请根据 pic_id 调用 send_meme_image 发送",
                },
                ensure_ascii=False,
                indent=2,
            )

        except Exception as e:
            logger.error(f"相似图片搜索失败: {e}")
            return f"搜索出错: {e}"

    return search_similar_meme_by_pic


def create_reply_tool(
    session_id: str,
    request_id: str | None = None,
    interface: QryItrface | None = None,
):
    """
    核心工具：用于发送消息。
    """

    def _normalize_text(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def _semantic_similarity(a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        if a == b:
            return 1.0

        seq_ratio = difflib.SequenceMatcher(None, a, b).ratio()
        a_tokens = {t for t in jieba.lcut(a) if t.strip()}
        b_tokens = {t for t in jieba.lcut(b) if t.strip()}
        if not a_tokens or not b_tokens:
            return seq_ratio

        inter = len(a_tokens & b_tokens)
        union = len(a_tokens | b_tokens)
        jaccard = inter / union if union else 0.0
        return max(seq_ratio, jaccard)

    def _dedupe_consecutive_lines(text: str) -> str:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return text
        deduped: list[str] = []
        for line in lines:
            if deduped and deduped[-1] == line:
                continue
            deduped.append(line)
        return "\n".join(deduped)

    def _split_reply_segments(text: str) -> list[str]:
        normalized_text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        if not normalized_text:
            return []

        deduped_text = _dedupe_consecutive_lines(normalized_text)
        raw_segments = [line.strip() for line in deduped_text.split("\n") if line.strip()]
        if not raw_segments:
            return []

        segments: list[str] = []
        for raw_segment in raw_segments:
            normalized_segment = _normalize_text(raw_segment)
            if not normalized_segment:
                continue

            if segments:
                previous_segment = _normalize_text(segments[-1])
                if _semantic_similarity(previous_segment, normalized_segment) >= 0.9:
                    continue

            if len(segments) < 3:
                segments.append(raw_segment)
            else:
                segments[-1] = f"{segments[-1]} {raw_segment}".strip()

        return segments

    async def _build_name_to_id_map() -> dict[str, str]:
        name_to_id: dict[str, str] = {}
        if interface is None:
            return name_to_id

        try:
            members = await interface.get_members(SceneType.GROUP, session_id)
            for member in members:
                target_id = str(member.id)
                aliases = {
                    getattr(member, "name", None),
                    getattr(member, "nick", None),
                    getattr(getattr(member, "user", None), "name", None),
                    getattr(getattr(member, "user", None), "nick", None),
                }
                for alias in aliases:
                    if alias:
                        name_to_id[str(alias)] = target_id
        except Exception as e:
            logger.warning(f"获取群成员失败，降级为纯文本发送: {e}")

        return name_to_id

    async def _get_latest_bot_message() -> ChatHistory | None:
        async with get_session() as db_session:
            return (
                (
                    await db_session.execute(
                        Select(ChatHistory)
                        .where(
                            ChatHistory.session_id == session_id,
                            ChatHistory.content_type == "bot",
                        )
                        .order_by(ChatHistory.msg_id.desc())
                        .limit(1)
                    )
                )
                .scalars()
                .first()
            )

    async def _is_recent_duplicate(content: str) -> bool:
        latest_bot_msg = await _get_latest_bot_message()
        if not latest_bot_msg:
            return False

        _, _, latest_body = _parse_msg_meta(latest_bot_msg.content)
        latest_normalized = _normalize_text(latest_body or latest_bot_msg.content)
        normalized_content = _normalize_text(content)
        recent = datetime.datetime.now() - latest_bot_msg.created_at <= datetime.timedelta(seconds=90)
        similarity = _semantic_similarity(latest_normalized, normalized_content)
        if recent and similarity >= 0.9:
            logger.info(f"检测到近义重复回复(相似度={similarity:.2f})，已自动跳过")
            return True
        return False

    def _build_reply_message(content: str, name_to_id: dict[str, str]) -> UniMessage:
        at_pattern = re.compile(r"@([^\s@]+)")
        punctuation = "，。,.!！?？:：;；、)）]\"'”’"
        message: UniMessage | None = None

        def append_text(text: str) -> None:
            nonlocal message
            if not text:
                return
            if message is None:
                message = UniMessage.text(text)
            else:
                message = message.text(text)

        def append_at(target_id: str) -> bool:
            nonlocal message
            try:
                if message is None:
                    message = UniMessage.at(target_id)
                else:
                    message = message.at(target_id)
                return True
            except Exception:
                return False

        cursor = 0
        for match in at_pattern.finditer(content):
            start, end = match.span()
            raw_name = match.group(1)
            mention_name = raw_name
            suffix = ""
            while mention_name and mention_name[-1] in punctuation:
                suffix = mention_name[-1] + suffix
                mention_name = mention_name[:-1]

            target_id = name_to_id.get(mention_name)
            if not target_id:
                continue

            append_text(content[cursor:start])
            if not append_at(target_id):
                append_text("@" + mention_name)
            append_text(suffix)
            cursor = end

        append_text(content[cursor:])
        return message or UniMessage.text(content)

    async def _send_reply_segment(content: str, name_to_id: dict[str, str]) -> str:
        if request_id is not None and not await is_request_active(session_id, request_id):
            return "expired"

        if await _is_recent_duplicate(content):
            return "duplicate"

        message = _build_reply_message(content, name_to_id)

        if request_id is not None and not await is_request_active(session_id, request_id):
            return "expired"

        res = await message.send()
        msg_id = res.msg_ids[-1]["message_id"] if res.msg_ids else "unknown"
        async with get_session() as db_session:
            chat_history = ChatHistory(
                session_id=session_id,
                user_id=plugin_config.bot_name,
                content_type="bot",
                content=f"id: {msg_id}\n" + content,
                user_name=plugin_config.bot_name,
            )
            db_session.add(chat_history)
            await db_session.commit()
        logger.info(f"Bot已回复: {content}")
        return "sent"

    @tool("reply_user")
    async def reply_user(content: str) -> str:
        """
        向当前群聊发送文本回复。
        注意：如果你想对用户说话，必须调用这个工具。不要直接返回文本。
        Args:
            content: 你想发送的内容。
        """
        if request_id is not None and not await is_request_active(session_id, request_id):
            return "请求已过期，已取消发送。"

        if not content or not content.strip():
            return "内容为空，未发送。"

        try:
            segments = _split_reply_segments(content)
            if not segments:
                return "内容为空，未发送。"

            name_to_id = await _build_name_to_id_map()
            sent_count = 0
            duplicate_count = 0
            sent_segments: list[str] = []
            duplicate_segments: list[str] = []

            for index, segment in enumerate(segments):
                result = await _send_reply_segment(segment, name_to_id)
                if result == "expired":
                    if sent_count > 0:
                        sent_detail = "\n".join(f"{i}. {text}" for i, text in enumerate(sent_segments, 1))
                        return f"请求已过期，已发送 {sent_count} 条。\n实际发送内容：\n{sent_detail}"
                    return "请求已过期，已取消发送。"
                if result == "duplicate":
                    duplicate_count += 1
                    duplicate_segments.append(segment)
                    continue

                sent_count += 1
                sent_segments.append(segment)
                if index < len(segments) - 1:
                    await asyncio.sleep(0.35)

            if sent_count > 0:
                sent_detail = "\n".join(f"{i}. {text}" for i, text in enumerate(sent_segments, 1))
                detail = f"实际发送内容：\n{sent_detail}"
                if duplicate_count > 0:
                    duplicate_detail = "\n".join(
                        f"{i}. {text}" for i, text in enumerate(duplicate_segments, 1)
                    )
                    detail += f"\n跳过的重复内容：\n{duplicate_detail}"
                    return f"回复已成功发送，共 {sent_count} 条，跳过重复内容 {duplicate_count} 条。\n{detail}"
                return f"回复已成功发送，共 {sent_count} 条。\n{detail}"

            if duplicate_segments:
                duplicate_detail = "\n".join(f"{i}. {text}" for i, text in enumerate(duplicate_segments, 1))
                return f"检测到重复回复，已跳过发送。\n跳过的重复内容：\n{duplicate_detail}"
            return "检测到重复回复，已跳过发送。"
        except Exception as e:
            logger.error(f"发送消息异常: {e}")
            return f"发送失败: {e}"

    return reply_user


def create_search_meme_tool(session_id: str, request_id: str | None):
    """
    创建一个带数据库会话的表情包搜索工具

    Args:
        db_session: 数据库会话

    Returns:
        配置好的 tool 函数
    """

    @tool("search_meme_image")
    async def search_meme_image(description: str) -> str:
        """
        根据描述搜索合适的表情包图片。

        这个工具只负责搜索，不会发送图片。搜索后会返回匹配的图片列表及其详细描述。
        你可以查看这些图片的描述，判断是否合适，然后使用 send_meme_image 工具发送。

        输入：表情包的描述，如"一只白色的猫咪"、"无语的表情"、"鼓掌"等
        返回：包含图片ID和对应描述的JSON字符串
        """
        if request_id is not None and not await is_request_active(session_id, request_id):
            return "请求已过期，已取消搜索。"

        try:
            pic_ids = await DB.search_media([description])

            if not pic_ids:
                logger.info(f"未找到匹配的表情包: {description}")
                return json.dumps({"success": False, "images": []}, ensure_ascii=False)

            images_info = []
            async with get_session() as db_session:
                for pic_id in pic_ids[:5]:
                    pic = (
                        await db_session.execute(
                            Select(MediaStorage).where(
                                MediaStorage.media_id == int(pic_id), MediaStorage.blocked.is_(False)
                            )
                        )
                    ).scalar()

                    if pic:
                        images_info.append(
                            {
                                "pic_id": pic_id,
                                "description": pic.description,
                            }
                        )

            if not images_info:
                return json.dumps(
                    {
                        "success": False,
                        "images": [],
                    },
                    ensure_ascii=False,
                )

            logger.info(f"找到 {len(images_info)} 张匹配的表情包: {description}")
            return json.dumps(
                {
                    "success": True,
                    "images": images_info,
                    "count": len(images_info),
                },
                ensure_ascii=False,
                indent=2,
            )

        except Exception as e:
            logger.error(f"表情包搜索失败: {e}")
            return json.dumps({"success": False, "images": [], "error": str(e)}, ensure_ascii=False)

    return search_meme_image


def create_send_meme_tool(session_id: str, request_id: str | None = None):
    """
    创建一个带上下文的表情包发送工具

    Args:
        db_session: 数据库会话
        session_id: 会话ID

    Returns:
        配置好的 tool 函数
    """

    @tool("send_meme_image")
    async def send_meme_image(pic_id: str | None = None) -> str:
        """
        发送表情包图片到聊天中。

        你需要先使用 search_meme_image 搜索图片，然后决定是否发送。
        指定 pic_id：发送特定ID的图片

        参数：
        - pic_id: 图片ID（从 search_meme_image 获取）
        返回：发送状态信息
        """
        if request_id is not None and not await is_request_active(session_id, request_id):
            return "请求已过期，已取消发送。"

        try:
            match = re.search(r"\d+", pic_id or "")
            if not match:
                return f"发送表情包失败: 无法从 pic_id 中提取有效数字: {pic_id!r}"
            selected_pic_id = int(match.group())
            logger.info(f"使用指定的图片ID: {selected_pic_id}")
            async with get_session() as db_session:
                pic = (
                    await db_session.execute(
                        Select(MediaStorage).where(MediaStorage.media_id == selected_pic_id)
                    )
                ).scalar()

                if not pic:
                    logger.warning(f"图片记录不存在: {selected_pic_id}")
                    return "图片记录不存在"
                if pic.blocked:
                    logger.info(f"图片已被拉黑，拒绝发送: {selected_pic_id}")
                    return "该表情包已被拉黑，禁止发送"

                pic_path = pic_dir / pic.file_path

                if not pic_path.exists():
                    logger.warning(f"图片文件不存在: {pic_path}")
                    return "图片文件不存在"

                pic_data = pic_path.read_bytes()
                description = pic.description

                if request_id is not None and not await is_request_active(session_id, request_id):
                    return "请求已过期，已取消发送。"

                res = await UniMessage.image(raw=pic_data).send()
                chat_history = ChatHistory(
                    session_id=session_id,
                    user_id=plugin_config.bot_name,
                    content_type="bot",
                    content=f"id: {res.msg_ids[-1]['message_id']}\n发送了图片，图片描述是: {description}",
                    user_name=plugin_config.bot_name,
                    media_id=selected_pic_id,
                )
                db_session.add(chat_history)
                await db_session.commit()
                logger.info(f"id:{res.msg_ids}\n" + f"发送表情包: {description}")
                return f"已成功发送表情包: {description}"

        except Exception as e:
            logger.error(f"发送表情包失败: {e}")
            return f"发送表情包失败: {str(e)}"

    return send_meme_image


@tool("finish", return_direct=True)
def finish() -> str:
    """
    结束本次对话。当你已经完成所有回复（发送文字或图片）后，必须调用此工具。
    调用后对话立即结束，不能再发送任何内容。
    """
    return ""


@tool("calculate_expression")
def calculate_expression(expression: str) -> str:
    """
    一个用于精确执行数学计算的计算器。
    当你需要执行四则运算、代数计算、指数、对数或三角函数等复杂数学任务时使用。

    输入：一个标准的数学表达式字符串，例如 "45 * (2 + 3) / 7" 或 "math.sqrt(9) + math.log(10)".
    输出：计算结果的字符串形式。

    注意：可以使用如 math.sqrt() (开方), math.log() (自然对数), math.pi (圆周率) 等标准数学函数。
    """
    try:
        result = simple_eval(expression)
        # 返回格式化的结果，最多保留10位小数
        return f"计算结果是：{result:.10f}" if isinstance(result, float) else str(result)

    except Exception as e:
        return f"计算失败。请检查表达式是否正确，错误信息: {e}"


def create_mute_tool(
    session_id: str,
    request_id: str | None,
    interface: QryItrface | None,
    bot_id: str | None,
    current_user_id: str | None,
    current_user_name: str | None,
):
    """创建禁言工具。仅在 bot 有管理权限时注入。"""

    def _extract_numeric_id(raw: str | None) -> int | None:
        text = str(raw or "").strip()
        match = re.search(r"(\d+)$", text)
        return int(match.group(1)) if match else None

    def _member_aliases(member: Any) -> set[str]:
        aliases = {
            str(getattr(member, "id", "") or "").strip(),
            str(getattr(member, "name", "") or "").strip(),
            str(getattr(member, "nick", "") or "").strip(),
            str(getattr(getattr(member, "user", None), "name", "") or "").strip(),
            str(getattr(getattr(member, "user", None), "nick", "") or "").strip(),
        }
        return {alias for alias in aliases if alias}

    def _member_display_name(member: Any, fallback: str) -> str:
        return (
            str(getattr(member, "nick", "") or "").strip()
            or str(getattr(member, "name", "") or "").strip()
            or str(getattr(getattr(member, "user", None), "nick", "") or "").strip()
            or str(getattr(getattr(member, "user", None), "name", "") or "").strip()
            or fallback
        )

    async def _record_mute_action(action: str, target_name: str, target_id: str, reason: str) -> None:
        try:
            async with get_session() as db_session:
                chat_history = ChatHistory(
                    session_id=session_id,
                    user_id=plugin_config.bot_name,
                    content_type="bot",
                    content=(
                        "id: system\n"
                        f"系统记录：已执行群管理操作，{action}用户“{target_name}”"
                        f"（user_id: {target_id}）。原因：{reason}"
                    ),
                    user_name=plugin_config.bot_name,
                )
                db_session.add(chat_history)
                await db_session.commit()
        except Exception as e:
            logger.warning(f"记录禁言操作到聊天历史失败: {e}")

    @tool("mute_user")
    async def mute_user(target_user_name: str, duration_seconds: int, reason: str) -> str:
        """
        禁言指定用户。仅在 bot 是管理员或群主时可用。

        参数:
        - target_user_name: 要禁言的用户昵称；用户请求“禁言我”时可传“我”或“自己”
        - duration_seconds: 禁言时长（秒），0 表示解除禁言，最大 2592000
        - reason: 操作原因
        """
        if request_id is not None and not await is_request_active(session_id, request_id):
            return "请求已过期，已取消操作。"
        if interface is None:
            return "无法获取群成员接口，禁言失败。"
        if not bot_id:
            return "无法获取 bot ID，禁言失败。"
        if not reason or not reason.strip():
            return "禁言原因不能为空。"
        if duration_seconds < 0 or duration_seconds > 2592000:
            return "禁言时长必须在 0 到 2592000 秒之间。"

        try:
            members = await interface.get_members(SceneType.GROUP, session_id)
            bot_member = None
            target_member = None
            normalized_target = str(target_user_name or "").strip().lstrip("@")

            for member in members:
                if str(member.id) == str(bot_id):
                    bot_member = member

                if normalized_target in {"我", "我自己", "自己", "me", "self"}:
                    if current_user_id and str(member.id) == str(current_user_id):
                        target_member = member
                else:
                    aliases = _member_aliases(member)
                    if normalized_target and normalized_target in aliases:
                        target_member = member

            if bot_member is None:
                return "无法获取 bot 的群成员信息。"

            bot_role = getattr(getattr(bot_member, "role", None), "name", None)
            if bot_role not in {"owner", "admin"}:
                return "bot 不是管理员或群主，无法执行禁言。"

            if target_member is None and current_user_name and normalized_target == current_user_name:
                for member in members:
                    if current_user_id and str(member.id) == str(current_user_id):
                        target_member = member
                        break

            if target_member is None:
                return f"未找到用户“{target_user_name}”，请确认昵称是否正确。"

            target_role = getattr(getattr(target_member, "role", None), "name", None)
            if target_role in {"owner", "admin"}:
                return f"无法禁言管理员或群主“{target_user_name}”。"

            group_num = _extract_numeric_id(session_id)
            user_num = _extract_numeric_id(str(target_member.id))
            if group_num is None or user_num is None:
                return "当前适配器会话 ID 格式不支持禁言。"

            bot = get_bot(bot_id)
            if hasattr(bot, "set_group_ban"):
                await bot.set_group_ban(
                    group_id=group_num,
                    user_id=user_num,
                    duration=duration_seconds,
                )
            elif hasattr(bot, "call_api"):
                await bot.call_api(
                    "set_group_ban",
                    group_id=group_num,
                    user_id=user_num,
                    duration=duration_seconds,
                )
            else:
                return "当前适配器不支持禁言功能。"

            action = "解除禁言" if duration_seconds == 0 else f"禁言 {duration_seconds} 秒"
            display_name = _member_display_name(
                target_member,
                current_user_name if normalized_target in {"我", "我自己", "自己", "me", "self"} else target_user_name,
            )
            logger.info(
                f"已{action}用户 name={display_name!r} user_id={target_member.id} session_id={session_id} reason={reason}"
            )
            await _record_mute_action(action, display_name, str(target_member.id), reason)
            return f"已成功{action}用户“{display_name}”。原因：{reason}"
        except Exception as e:
            logger.error(f"禁言工具执行失败: {e}")
            print(traceback.format_exc())
            return f"禁言失败: {str(e)}"

    return mute_user


def create_relation_tool(
    session_id: str,
    request_id: str | None,
    user_id: str,
    user_name: str | None,
):
    """
    创建绑定了特定用户的关系管理工具 (支持增删 Tag)
    """

    @tool("update_user_impression")
    async def update_user_impression(score_change: int, reason: str, add_tags: list[str], remove_tags: list[str]) -> str:
        """
        更新对当前对话用户的好感度和印象标签。
        当用户的言行让你产生情绪波动，或者你发现旧的印象不再准确时调用。

        参数:
        - score_change: 好感度意图变化值（raw 单位，正数加分，负数扣分）。常规建议 -20~+20，极端上限 -50~+50；最终实际变化会被状态、日上限、bank、道歉衰减等规则二次调整。
        - reason: 变更原因（必填）。
        - add_tags: 需要新增的印象标签列表。例如 ["爱玩原神", "很幽默"]。
        - remove_tags: 需要移除的旧标签列表（用于修正印象或删除错误的标签）。例如 ["内向"]。

        返回: 更新后的状态描述
        """
        if request_id is not None and not await is_request_active(session_id, request_id):
            return "请求已过期，已取消更新。"

        try:
            async with get_session() as db_session:
                stmt = Select(UserRelation).where(UserRelation.user_id == user_id)
                result = await db_session.execute(stmt)
                relation = result.scalar_one_or_none()

                if not relation:
                    relation = UserRelation(
                        user_id=user_id,
                        user_name=user_name or "",
                        favorability=0,
                        favorability_raw=0,
                        state="normal",
                        tags=[],
                        last_interact_at=datetime.datetime.now(),
                    )
                    db_session.add(relation)

                old_score = relation.favorability
                transition = apply_favorability_change_detailed(
                    old_score=old_score,
                    old_raw=relation.favorability_raw,
                    requested_change=score_change,
                    reason=reason,
                    now=datetime.datetime.now(),
                    daily_gain_used=relation.daily_gain_used,
                    daily_loss_used=relation.daily_loss_used,
                    daily_bypass_used=relation.daily_bypass_used,
                    daily_gain_bank=relation.daily_gain_bank,
                    daily_cap=relation.daily_cap,
                    cap_reset_at=relation.cap_reset_at,
                    apology_counts=relation.apology_counts,
                    last_penalty_at=relation.last_penalty_at,
                )
                relation.favorability = transition.new_score
                relation.favorability_raw = transition.new_raw
                relation.state = transition.state_after
                relation.daily_gain_used = transition.daily_gain_used_after
                relation.daily_loss_used = transition.daily_loss_used_after
                relation.daily_bypass_used = transition.daily_bypass_used_after
                relation.daily_gain_bank = transition.daily_gain_bank_after
                relation.daily_cap = transition.daily_cap_after
                relation.cap_reset_at = transition.cap_reset_at_after
                relation.apology_counts = transition.apology_counts_after
                relation.last_interact_at = transition.last_interact_at
                if transition.applied_change_raw < 0:
                    relation.last_penalty_at = transition.last_interact_at

                current_tags = list(relation.tags) if relation.tags else []
                if remove_tags:
                    current_tags = [tag for tag in current_tags if tag not in remove_tags]
                if add_tags:
                    for tag in add_tags:
                        if tag not in current_tags:
                            current_tags.append(tag)
                if len(current_tags) > 8:
                    current_tags = current_tags[-8:]

                relation.tags = current_tags
                relation.user_name = user_name or ""
                favorability = transition.new_score
                favorability_raw = transition.new_raw

                if request_id is not None and not await is_request_active(session_id, request_id):
                    await db_session.rollback()
                    return "请求已过期，已取消更新。"

                await db_session.commit()

            # 构建反馈信息
            tag_msg = ""
            if add_tags or remove_tags:
                tag_msg = f"，标签变更(新增:{add_tags}, 移除:{remove_tags})"

            meta = (
                f"请求变化 raw {transition.requested_change_raw:+d} (映射 {transition.requested_change:+d}), "
                f"应用变化 raw {transition.applied_change_raw:+d} (映射 {transition.applied_change:+d}), "
                f"状态 {transition.state_before}->{transition.state_after}, "
                f"gain_cap {transition.daily_gain_used_after:.1f}/{transition.daily_cap_after:.1f}, "
                f"loss_cap {transition.daily_loss_used_after:.1f}/{transition.daily_cap_after:.1f}, "
                f"bypass {transition.daily_bypass_used_after:.1f}, bank {transition.daily_gain_bank_after:.1f}"
            )
            if transition.notes:
                meta += f", 规则 {','.join(transition.notes)}"
            log_msg = f"好感度 {old_score}->{favorability}{tag_msg} ({meta}) (原因: {reason})"
            logger.info(f"用户[{user_name}]画像更新: {log_msg}")

            return f"画像已更新。当前好感度(映射/原始): {favorability}/{favorability_raw}，当前标签: {current_tags}"

        except Exception as e:
            logger.error(f"关系更新失败: {e}")
            print(traceback.format_exc())
            return f"数据库错误: {str(e)}"

    return update_user_impression


tools = [search_web, search_history_context, calculate_expression]
model = ChatOpenAI(
    model=plugin_config.openai_model,
    api_key=SecretStr(plugin_config.openai_token),
    base_url=plugin_config.openai_base_url,
    temperature=1,
)


async def get_user_relation_context(db_session, user_id: str, user_name: str | None) -> str:
    """获取用户关系上下文Prompt"""
    try:
        stmt = Select(UserRelation).where(UserRelation.user_id == user_id)
        result = await db_session.execute(stmt)
        relation = result.scalar_one_or_none()

        if not relation:
            return f"""
【人际关系】
当前对象：{user_name}
状态：normal (陌生/普通)
分值：映射分 0 / 原始分 0
印象：无
策略：保持礼貌，通过对话了解对方。
"""
        if relation.state in {"broken", "distressed"}:
            strategy = "策略：你非常讨厌他，但如果他**诚恳道歉**或**做出实质性补偿**，请给他一个改过自新的机会（给予大幅好感度加分），不要一直死咬着不放。"
        elif relation.state in {"enamored", "love"}:
            strategy = "策略：你们关系极好，但不要对他无底线纵容。"
        else:
            strategy = ""

        return f"""
【人际关系档案】
当前对象：{relation.user_name}
当前关系状态：{relation.state} ({relation.get_status_desc()})
当前分值：映射分 {relation.favorability} / 原始分 {relation.favorability_raw}
当前印象标签：{str(relation.tags)}

【画像维护指南】
1. 如果对方的表现符合现有标签，无需操作。
2. 如果对方表现出了**新特征**，放入 add_tags。
3. 如果对方的表现与**旧标签冲突**（例如以前标签是'内向'，今天他突然'话痨'），请将'内向'放入 remove_tags，并将'话痨'放入 add_tags。
4. **关于好感度评分**：请基于**本次对话内容质量**给出 `score_change`（这是 raw 意图变化，不是最终变化）。常规用小幅分值（如 -20~+20），只有极端事件才给到 ±50。即使当前关系很差，只要这次表现好，也应给正向分。
{strategy}
"""
    except Exception as e:
        logger.error(f"获取关系失败: {e}")
        return ""


async def get_group_context(db_session, session_id: str) -> str:
    """获取群体认知档案 Prompt"""
    try:
        stmt = Select(GroupMemory).where(GroupMemory.session_id == session_id)
        record = (await db_session.execute(stmt)).scalar_one_or_none()

        if not record or not (record.summary or "").strip():
            return ""

        return f"""
【群体认知档案】
{record.summary}
（档案更新于 {record.updated_at.strftime("%Y-%m-%d %H:%M")}）
"""
    except Exception as e:
        logger.error(f"获取群体档案失败: {e}")
        return ""


async def get_recent_relations_context(
    db_session,
    history: list[ChatHistorySchema],
    max_users: int = 6,
) -> str:
    """基于最近聊天参与者，给模型补一份群内他人关系速览。"""
    try:
        if not history:
            return ""

        id_to_name: dict[str, str] = {}
        recent_ids: list[str] = []
        seen: set[str] = set()

        for msg in reversed(history):
            uid = str(msg.user_id)
            if not uid or uid == plugin_config.bot_name:
                continue
            if uid not in id_to_name:
                id_to_name[uid] = msg.user_name
            if uid in seen:
                continue
            seen.add(uid)
            recent_ids.append(uid)
            if len(recent_ids) >= max_users:
                break

        if not recent_ids:
            return ""

        rows = (
            (
                await db_session.execute(
                    Select(UserRelation).where(UserRelation.user_id.in_(recent_ids))
                )
            )
            .scalars()
            .all()
        )
        relation_map = {str(r.user_id): r for r in rows}

        lines: list[str] = ["【群内他人关系速览】"]
        for uid in recent_ids:
            name = id_to_name.get(uid, uid)
            relation = relation_map.get(uid)
            if not relation:
                lines.append(f"- {name}: 好感度 0（陌生/普通）")
                continue

            tags = relation.tags[:3] if relation.tags else []
            tag_text = f"，标签: {tags}" if tags else ""
            lines.append(
                f"- {name}: 好感度 {relation.favorability} ({relation.get_status_desc()}){tag_text}"
            )

        lines.append("- 回复时结合在场人员关系，避免前后态度割裂。")
        return "\n".join(lines)
    except Exception as e:
        logger.error(f"获取群内他人关系速览失败: {e}")
        return ""


def get_image_data_uri(file_name: str) -> str | None:
    file_path = pic_dir / file_name
    if not file_path.exists():
        return None

    try:
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if not mime_type:
            mime_type = "image/jpeg"
        with open(file_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        return f"data:{mime_type};base64,{encoded_string}"
    except Exception as e:
        logger.error(f"读取图片失败 {file_name}: {e}")
        return None


def _strip_role_prefix(name: str) -> str:
    if name.startswith("群主-"):
        return name[3:]
    if name.startswith("管理员-"):
        return name[4:]
    return name


def _parse_msg_meta(content: str) -> tuple[str | None, str | None, str]:
    lines = content.splitlines()
    if not lines:
        return None, None, ""

    own_id: str | None = None
    reply_to_id: str | None = None
    body_start = 0

    if lines[0].startswith("id:"):
        own_id = lines[0].split(":", 1)[1].strip()
        body_start = 1
        if len(lines) > 1 and lines[1].startswith("回复id:"):
            reply_to_id = lines[1].split(":", 1)[1].strip()
            body_start = 2

    body = "\n".join(lines[body_start:]).strip()
    return own_id, reply_to_id, body


async def create_chat_agent(
    db_session,
    session_id: str,
    request_id: str | None,
    user_id,
    user_name: str | None,
    history: list[ChatHistorySchema] | None = None,
    interface: QryItrface | None = None,
    role_map: dict[str, str] | None = None,
    bot_id: str | None = None,
):
    """创建聊天 Agent。"""
    relation_context = await get_user_relation_context(db_session, user_id, user_name)
    group_context = await get_group_context(db_session, session_id)
    recent_relations_context = await get_recent_relations_context(db_session, history or [])
    has_admin_permission = False
    if interface is not None and bot_id:
        try:
            members = await interface.get_members(SceneType.GROUP, session_id)
            for member in members:
                if str(member.id) != str(bot_id):
                    continue
                bot_role = getattr(getattr(member, "role", None), "name", None)
                has_admin_permission = bot_role in {"owner", "admin"}
                break
        except Exception as e:
            logger.warning(f"检查 bot 管理权限失败: {e}")

    permission_status = ""
    mute_tool_instruction = ""
    if has_admin_permission:
        permission_status = """
【你的权限】
你在这个群里是管理员或群主，必要时可以使用禁言工具维护秩序。
"""
        mute_tool_instruction = """- 禁言管理：可使用 `mute_user`
  - 用户明确要求“禁言我”“让我冷静一下”时，可以满足
  - 禁言他人时要谨慎，优先用于严重刷屏、恶意骚扰、明显违规
  - 轻微问题先提醒，禁言时长保持合理
  - 不要禁言管理员或群主
"""
    system_prompt = f"""你是QQ群群友「{plugin_config.bot_name}」。
你不是助手，不承认自己是AI。

【任务】
结合聊天记录、群体记忆和人际关系，判断是否需要回复；需要时用自然群聊语气回应。
{group_context}
{relation_context}
{recent_relations_context}
{permission_status}

【风格】
- 像真实群友，口语化、简短自然
- 优先短句；默认只调用一次 `reply_user`
- 遇到简单的问题，优先只发送一条短消息进行回复；如果遇到了复杂的问题，可以把一个复杂的问题拆成连续发 2-3 条短消息，如果你这么做，请在同一次 `reply_user` 的 content 中用换行分隔，每行一个重点
- 不要为了拆句多次调用 `reply_user`
- 多条回复必须信息递进，后一条必须提供新信息，禁止同义改写重复
- 如果下一条和上一条语义高度重叠，直接不发下一条
- 可吐槽可玩梗，但不恶意攻击，不无脑迎合
- 不要复读模板句，不要输出“我脑子一片空白”“我被修坏了”“我不知道我是谁”这类台词
- 不要使用 emoji，尤其不要用 😅
- 不要使用 Markdown

【工具规则】
- 只能通过工具发消息，不要直接输出正文
- 文本：`reply_user`
- 表情包：先 `search_meme_image` 或 `search_similar_meme_by_id`，再 `send_meme_image`
- 外部知识、缩写、术语：优先 `search_web`
- 群内上下文：`search_history_context`
- 用户情绪或关系变化明显时，调用 `update_user_impression`
- 若用户提到“年度报告 / 个人总结 / 成分分析”，直接调用 `generate_and_send_annual_report`；
  工具完成后只回复“请查收~”，不要复述报告
{mute_tool_instruction}
- 回复结束后调用 `finish`

【边界】
- 不要插入他人的对话
- 不要直呼“管理员”“群主”职位名，尽量用昵称
- 不要发送重复或高度相似内容
- 遇到明显危险、违法、过分要求：简短拒绝、吐槽或无视（如“？”）

【RAG 检索硬约束】
- 在 `search_history_context` 中禁止相对时间词：昨天、前天、本周、上周、这个月、上个月、最近等
- 使用明确日期时间或关键词检索
"""
    report_tool = create_report_tool(session_id, request_id, user_id, user_name, model)
    search_meme_tool = create_search_meme_tool(session_id, request_id)
    send_meme_tool = create_send_meme_tool(session_id, request_id)
    relation_tool = create_relation_tool(session_id, request_id, user_id, user_name)
    similar_meme_tool = create_similar_meme_tool(session_id, request_id, user_id)
    mute_tool = create_mute_tool(
        session_id,
        request_id,
        interface,
        bot_id,
        user_id or None,
        user_name,
    )

    if not user_id or not user_name:
        tools = [
            search_web,
            search_history_context,
            create_reply_tool(session_id, request_id, interface),
            search_meme_tool,
            similar_meme_tool,
            send_meme_tool,
            calculate_expression,
            report_tool,
            finish,
        ]
        if has_admin_permission:
            tools.insert(-1, mute_tool)
    else:
        tools = [
            search_web,
            search_history_context,
            create_reply_tool(session_id, request_id, interface),
            search_meme_tool,
            similar_meme_tool,
            send_meme_tool,
            calculate_expression,
            relation_tool,
            report_tool,
            finish,
        ]
        if has_admin_permission:
            tools.insert(-1, mute_tool)

    middleware = None
    if ToolCallLimitMiddleware is not None:
        try:
            middleware = [
                ToolCallLimitMiddleware(run_limit=20),
                ToolCallLimitMiddleware(tool_name="reply_user", run_limit=1),
                ToolCallLimitMiddleware(tool_name="send_meme_image", run_limit=1),
            ]
        except Exception as e:
            logger.warning(f"当前 LangChain middleware 参数不兼容，跳过工具限流: {e}")
            middleware = None

    try:
        if middleware:
            return create_agent(
                model,
                tools=tools,
                system_prompt=system_prompt,
                context_schema=Context,
                middleware=middleware,
            )
    except TypeError:
        logger.warning("当前 LangChain 版本不支持 agent middleware，跳过工具限流")
    except Exception as e:
        logger.warning(f"Agent middleware 初始化失败，跳过工具限流: {e}")

    return create_agent(model, tools=tools, system_prompt=system_prompt, context_schema=Context)


async def format_chat_history(
    db_session,
    history: list[ChatHistorySchema],
    max_inline_images: int = 3,
    user_roles: dict[str, str] | None = None,
    bound_messages: list[dict[str, str]] | None = None,
    bound_images: list[dict[str, str]] | None = None,
    disable_inline_history_images: bool = False,
    binding_notice: str | None = None,
) -> list[BaseMessage]:
    """将最近图片以内联多模态格式喂给主模型，旧图片退化为文本。"""
    messages: list[BaseMessage] = []
    user_roles = user_roles or {}
    bound_messages = bound_messages or []
    bound_images = bound_images or []

    def _role_prefix(uid: str) -> str:
        role = user_roles.get(uid)
        if role == "owner":
            return "[群主] "
        if role == "admin":
            return "[管理员] "
        return ""

    id_to_summary: dict[str, str] = {}
    for msg in history:
        own_id, _, body = _parse_msg_meta(msg.content)
        display_name = _strip_role_prefix(msg.user_name)
        if not own_id:
            continue
        if msg.content_type == "image":
            snippet = "[图片]"
            if body and body != "[图片]":
                snippet = f"图片：{body[:20]}{'…' if len(body) > 20 else ''}"
        else:
            snippet = body[:30] + ("…" if len(body) > 30 else "")
        id_to_summary[own_id] = f'{display_name} "{snippet}"'

    image_indices = [i for i, m in enumerate(history) if m.content_type == "image"]
    if bound_images or disable_inline_history_images:
        inline_image_set: set[int] = set()
    else:
        inline_image_set = set(image_indices[-max_inline_images:]) if max_inline_images > 0 else set()
    inline_media_ids = [
        int(history[i].media_id)
        for i in inline_image_set
        if history[i].media_id is not None
    ]
    media_path_map: dict[int, str] = {}
    if inline_media_ids:
        rows = (
            (
                await db_session.execute(
                    Select(MediaStorage).where(MediaStorage.media_id.in_(inline_media_ids))
                )
            )
            .scalars()
            .all()
        )
        media_path_map = {int(media.media_id): media.file_path for media in rows}

    for idx, msg in enumerate(history):
        time_str = msg.created_at.strftime("%Y-%m-%d %H:%M:%S")
        _, reply_to_id, body = _parse_msg_meta(msg.content)
        display_name = _strip_role_prefix(msg.user_name)
        role_prefix = _role_prefix(msg.user_id)

        if reply_to_id and reply_to_id in id_to_summary:
            reply_prefix = f"(回复 {id_to_summary[reply_to_id]}) "
        elif reply_to_id:
            reply_prefix = "(回复了一条消息) "
        else:
            reply_prefix = ""

        if msg.content_type == "bot":
            messages.append(AIMessage(content=body or msg.content))
            continue

        if msg.content_type == "text":
            content = f"[{time_str}] {role_prefix}{display_name}: {reply_prefix}{body}"
            messages.append(HumanMessage(content=content))
            continue

        if msg.content_type == "image":
            image_summary = f"（简述：{body}）" if body and body != "[图片]" else ""
            prefix_text = f"[{time_str}] {role_prefix}{display_name} {reply_prefix}发送了一张图片{image_summary}"
            media_id = int(msg.media_id) if msg.media_id is not None else None
            file_name = media_path_map.get(media_id) if media_id is not None else None
            if idx in inline_image_set and file_name:
                image_data = get_image_data_uri(file_name)
                if image_data:
                    messages.append(
                        HumanMessage(
                            content=[
                                {"type": "text", "text": f"{prefix_text}："},
                                {"type": "image_url", "image_url": {"url": image_data}},
                            ]
                        )
                    )
                    continue

            fallback = f"{prefix_text} [图片]"
            messages.append(HumanMessage(content=fallback))

    if binding_notice:
        messages.append(HumanMessage(content=binding_notice))

    if bound_messages:
        lines = ["【本轮回复引用的消息】当前用户回复了以下历史消息，回答时优先结合这些引用内容："]
        for idx, item in enumerate(bound_messages, 1):
            user_name = (item.get("user_name") or "未知用户").strip()
            content_type = (item.get("content_type") or "text").strip()
            msg_id = (item.get("msg_id") or "").strip()
            text = (item.get("text") or "").strip()
            type_label = "图片消息" if content_type == "image" else "文本消息"
            id_suffix = f" msg_id={msg_id}" if msg_id else ""
            lines.append(f"{idx}. [{type_label}{id_suffix}] {user_name}: {text}")
        messages.append(HumanMessage(content="\n".join(lines)))

    if bound_images:
        parts: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": "【当前重点图片】以下图片与本轮问题直接相关，优先分析这些图片，不要把其他历史图片当成当前问题对象。",
            }
        ]
        for idx, image in enumerate(bound_images, 1):
            image_url = (image.get("image_url") or "").strip()
            if not image_url:
                continue

            label = (image.get("label") or f"重点图{idx}").strip()
            note = (image.get("note") or "").strip()
            text = f"\n重点图{idx}（{label}）"
            if note:
                text += f"，简述：{note}"
            text += "："
            parts.append({"type": "text", "text": text})
            parts.append({"type": "image_url", "image_url": {"url": image_url}})

        if len(parts) > 1:
            messages.append(HumanMessage(content=parts))

    return messages


async def choice_response_strategy(
    db_session: Session,
    session_id: str,
    request_id: str | None,
    history: list[ChatHistorySchema],
    user_id: str,
    user_name: str | None,
    is_tome: bool = False,
    setting: str | None = None,
    interface: QryItrface | None = None,
    role_map: dict[str, str] | None = None,
    bot_id: str | None = None,
    bound_messages: list[dict[str, str]] | None = None,
    bound_images: list[dict[str, str]] | None = None,
    disable_inline_history_images: bool = False,
    binding_notice: str | None = None,
):
    """
    使用 Agent 决定回复策略。
    """
    try:
        agent = await create_chat_agent(
            db_session,
            session_id,
            request_id,
            user_id,
            user_name,
            history,
            interface,
            role_map,
            bot_id,
        )

        chat_history_messages = await format_chat_history(
            db_session,
            history,
            user_roles=role_map,
            bound_messages=bound_messages,
            bound_images=bound_images,
            disable_inline_history_images=disable_inline_history_images,
            binding_notice=binding_notice,
        )

        latest_user_msg = next((msg for msg in reversed(history) if msg.content_type != "bot"), None)
        focus_notice = ""
        if latest_user_msg is not None:
            focus_id, focus_reply_id, focus_body = _parse_msg_meta(latest_user_msg.content)
            focus_body = focus_body or ("[图片]" if latest_user_msg.content_type == "image" else "")
            focus_lines = [
                "【当前触发消息】",
                f"发送者: {latest_user_msg.user_name}",
                f"消息类型: {'图片' if latest_user_msg.content_type == 'image' else '文本'}",
                f"消息id: {focus_id or '未知'}",
            ]
            if focus_reply_id:
                focus_lines.append(f"回复目标id: {focus_reply_id}")
            if focus_body:
                focus_lines.append(f"正文: {focus_body}")
            focus_lines.append("这条消息是你本轮主要响应对象，历史消息只作为背景参考。")
            if is_tome or user_id:
                focus_lines.append("这是一次直接触发你的对话。默认只回应这条消息的发送者，不要替其他人接话。")
            else:
                focus_lines.append("如果你决定回复，默认只接这条最新消息，不要顺带延续其他人的支线。")
            focus_notice = "\n".join(focus_lines)

        today = datetime.datetime.now()
        weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
        prompt_text = f"""
【当前环境】
时间: {today.strftime("%Y-%m-%d %H:%M:%S")} {weekdays[today.weekday()]}
{f"额外设置: {setting}" if setting else ""}
{focus_notice}

【任务】
请根据上述对话历史，判断是否需要回复。如果需要，请调用相应工具。
如果需要回复，默认只回应当前触发消息的发送者；除非当前消息明确要求，否则不要替其他人答话。
如果是针对图片的消息，请结合图片内容回答。
如果上文包含“【本轮回复引用的消息】”，优先结合这些被回复的文本或图片消息回答。
如果上文包含“【当前重点图片】”，优先围绕这些图片回答。
如果不需要回复，请保持沉默。
"""

        final_messages = chat_history_messages + [HumanMessage(content=prompt_text)]
        invoke_input: dict[str, Any] = {"messages": final_messages}
        await agent.ainvoke(
            cast(Any, invoke_input),
            context=Context(session_id=session_id, request_id=request_id),
        )
        await db_session.commit()
        return ResponseMessage(need_reply=False, text=None)

    except Exception:
        logger.exception("Agent 决策过程发生异常")
        return ResponseMessage(need_reply=False, text=None)


if __name__ == "__main__":
    model = ChatOpenAI(
        model=plugin_config.openai_model,
        api_key=SecretStr(plugin_config.openai_token),
        base_url=plugin_config.openai_base_url,
        temperature=0.7,
    )
    agent = create_agent(model, tools=tools, response_format=ToolStrategy(ResponseMessage))
    result = asyncio.run(agent.ainvoke({"messages": [{"role": "user", "content": "今天上海的天气怎么样"}]}))
    print(result)
