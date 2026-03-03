import collections
import json
import asyncio
import datetime
import random
import re
import traceback
from typing import Any, cast
from pathlib import Path
from dataclasses import dataclass

import jieba
from langchain_core.prompts import ChatPromptTemplate
from nonebot import require, get_plugin_config
from pydantic import Field, BaseModel, SecretStr, field_validator
from simpleeval import simple_eval
from sqlalchemy import Select, func, extract, desc
from PIL import Image as PILImage

from nonebot.log import logger
from langchain.tools import ToolRuntime, tool
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from nonebot_plugin_alconna import UniMessage
from sqlalchemy.orm.session import Session
from langchain_core.messages import AIMessage, HumanMessage
from langchain.agents.structured_output import ToolStrategy

from ..model import ChatHistory, MediaStorage, UserRelation, ChatHistorySchema
from ..config import Config
from ..favorability import apply_monika_favorability_change
from ..milvus import MilvusOP

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


# 如果想封装成自定义的 @tool，可以这样写:
@tool("search_web")
async def search_web(query: str) -> str:
    """
    用于搜索最新的实时信息。当你需要最新的事实信息、天气或新闻时使用。
    输入：需要搜索的内容。
    """
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
    try:
        logger.info(f"大模型执行{runtime.context.session_id} RAG 搜索\n{query}")

        # Fast probe to avoid hanging the agent when Milvus (or the tunnel) is unavailable.
        try:
            client = MilvusOP._get_async_client()
            try:
                await asyncio.wait_for(client.list_collections(), timeout=2.0)
            except AttributeError:
                await asyncio.wait_for(client.has_collection(collection_name="chat_collection"), timeout=2.0)
        except Exception as e:
            logger.error(f"RAG skipped (Milvus unavailable): {e}")
            return '未找到相关历史记录'

        similar_msgs = await asyncio.wait_for(MilvusOP.search([query], search_filter=f'session_id == "{runtime.context.session_id}"'), timeout=8.0)
        return "\n".join(similar_msgs) if similar_msgs else "未找到相关历史记录"

    except asyncio.TimeoutError:
        logger.error("RAG search timed out; skipped")
        return '未找到相关历史记录'
    except Exception as e:
        logger.error(f"历史搜索失败: {e}")
        return '未找到相关历史记录'


def create_report_tool(db_session, session_id: str, user_id: str, user_name: str | None, llm_client: ChatOpenAI):
    """
    创建年度报告工具（限制在当前群聊 session_id 范围内）
    """

    @tool("generate_and_send_annual_report")
    async def generate_and_send_annual_report() -> str:
        """
        生成并发送当前群聊的年度报告。
        包含：个人在本群的统计、性格分析、全群排行榜以及Bot的好感度回顾。
        """
        try:
            logger.info(f"开始生成用户 {user_name} 在群 {session_id} 的年度报告...")
            now = datetime.datetime.now()
            current_year = now.year

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
                # 1. 第一步：只根据 user_id 进行分组统计
                # 注意：Select 里先不要查 user_name，因为我们还没有聚合它
                stmt = Select(ChatHistory.user_id, func.count(ChatHistory.msg_id).label("c")).where(extract("year", ChatHistory.created_at) == current_year, ChatHistory.session_id == session_id)

                if content_type:
                    stmt = stmt.where(ChatHistory.content_type == content_type)
                if hour_limit:
                    stmt = stmt.where(extract("hour", ChatHistory.created_at) < hour_limit)

                # 核心修改：只 group_by user_id
                stmt = stmt.group_by(ChatHistory.user_id).order_by(desc("c")).limit(3)

                # 获取结果，此时是 List[(user_id, count)]
                rows = (await db_session.execute(stmt)).all()

                if not rows:
                    return "虚位以待"

                # 2. 第二步：获取这些卷王的“最新昵称”
                # 因为只取前3名，这里循环查3次数据库完全没问题，且能保证昵称是最新的
                rank_items = []
                for uid, count in rows:
                    # 查询该用户最近的一条消息记录，取当时的名字
                    name_stmt = Select(ChatHistory.user_name).where(ChatHistory.user_id == uid).order_by(desc(ChatHistory.created_at)).limit(1)

                    latest_name = (await db_session.execute(name_stmt)).scalar()

                    # 兜底：如果查不到名字（极少情况），用 ID 代替
                    display_name = latest_name if latest_name else f"用户{uid}"
                    rank_items.append(f"{display_name}({count})")
                return ", ".join(rank_items)

            rank_talk = await get_rank_str()
            rank_img = await get_rank_str(content_type="image")
            rank_night = await get_rank_str(hour_limit=5)

            # 只分析本群的文本
            stmt_text = (
                Select(ChatHistory.content)
                .where(
                    ChatHistory.session_id == session_id,  # <--- 关键修改
                    extract("year", ChatHistory.created_at) == current_year,
                    ChatHistory.user_id == user_id,
                    ChatHistory.content_type == "text",
                )
                .order_by(desc(ChatHistory.created_at))
                .limit(2000)
            )  # 取本群最近2000条

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

            # 格式化关系描述，喂给 LLM
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
            await UniMessage.text(final_report_text).send()

            return "报告已生成并发送。"

        except Exception as e:
            logger.error(f"内部 LLM 生成报告失败: {e}")
            import traceback

            traceback.print_exc()
            return f"生成过程出错: {e}"

    return generate_and_send_annual_report


def create_similar_meme_tool(db_session, session_id: str):
    """
    创建基于消息ID搜索相似表情包的工具
    """

    @tool("search_similar_meme_by_id")
    async def search_similar_meme_by_pic() -> str:
        """
        根据指定的历史最新图片，搜索与之相似的表情包。
        当用户说“找一张跟这张差不多的”或引用某张图片求相似图时使用。
        """
        # 1. 清理 ID (防止模型传入 'id:12345' 这种格式)

        logger.info(f"正在搜索相似图片...")

        try:
            # 2. 从 ChatHistory 查找该消息的描述
            # 注意：这里假设 ChatHistory.msg_id 存的是平台的消息ID
            stmt = Select(ChatHistory).where(ChatHistory.session_id == session_id, ChatHistory.content_type == "image").order_by(desc(ChatHistory.created_at)).limit(1)
            result = await db_session.execute(stmt)
            msg = result.scalar_one_or_none()

            if not msg:
                return f"未找到历史消息。"

            if msg.content_type != "image":
                # 如果不是图片，尝试用文本内容去搜图片（也是一种玩法）
                description = msg.content
                pic_ids = await MilvusOP.search_media([description])
            else:
                # 如果是图片，ChatHistory.content 存的就是描述
                description = msg.content
                stmt = Select(MediaStorage).where(
                    MediaStorage.media_id == msg.media_id,
                )
                result = await db_session.execute(stmt)
                msg = result.scalar_one_or_none()
                pic_ids = await MilvusOP.search_media_by_pic([str(pic_dir / msg.file_path)])
            if not pic_ids:
                logger.info(f"未找到匹配的表情包: {description}")
                return "没有搜索到相似图片"
            # 从数据库获取每张图片的详细信息
            images_info = []
            for pic_id in pic_ids[:5]:  # 只返回前5张，避免信息过多
                pic = (
                    await db_session.execute(
                        Select(MediaStorage).where(MediaStorage.media_id == int(pic_id), MediaStorage.blocked.is_(False))
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
                logger.info(f"相似图检索结果均被黑名单过滤: {description}")
                return "没有搜索到相似图片"

            return json.dumps(
                {
                    "success": True,
                    "source_description": description,  # 告诉模型原图是啥
                    "images": images_info,
                    "count": len(images_info),
                },
                ensure_ascii=False,
                indent=2,
            )

        except Exception as e:
            logger.error(f"相似图片搜索失败: {e}")
            return f"搜索出错: {e}"

    return search_similar_meme_by_pic


def create_reply_tool(db_session, session_id: str):
    """
    核心工具：用于发送消息。
    """

    @tool("reply_user")
    async def reply_user(content: str) -> str:
        """
        向当前群聊发送文本回复。
        注意：如果你想对用户说话，必须调用这个工具。不要直接返回文本。
        Args:
            content: 你想发送的内容。
        """
        if content == "OVER":
            return "不要通过这个函数结束对话"

        if not content or not content.strip():
            return "内容为空，未发送。"

        try:
            # 1. 实际发送消息 (Side Effect)
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
            logger.info(f"Bot已回复: {content}")
            return "回复已成功发送，如果没有新的、不同的内容要补充，请立即结束回复(输出 OVER)。"
        except Exception as e:
            logger.error(f"发送消息异常: {e}")
            await db_session.rollback()
            return f"发送失败: {e}"

    return reply_user


def create_search_meme_tool(db_session):
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
        try:
            pic_ids = await MilvusOP.search_media([description])

            if not pic_ids:
                logger.info(f"未找到匹配的表情包: {description}")
                return json.dumps({"success": False, "images": []}, ensure_ascii=False)

            # 从数据库获取每张图片的详细信息
            images_info = []
            for pic_id in pic_ids[:5]:  # 只返回前5张，避免信息过多
                pic = (
                    await db_session.execute(
                        Select(MediaStorage).where(MediaStorage.media_id == int(pic_id), MediaStorage.blocked.is_(False))
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


def create_send_meme_tool(db_session, session_id: str):
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
        try:
            selected_pic_id = None
            if pic_id:
                selected_pic_id = int(pic_id)
                logger.info(f"使用指定的图片ID: {pic_id}")
            if not selected_pic_id:
                return "没有指定图片id"

            # 从数据库获取图片信息
            pic = (await db_session.execute(Select(MediaStorage).where(MediaStorage.media_id == int(selected_pic_id)))).scalar()

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

            # 读取图片数据
            pic_data = pic_path.read_bytes()
            description = pic.description
            # 发送图片
            res = await UniMessage.image(raw=pic_data).send()
            # 记录发送历史
            chat_history = ChatHistory(
                session_id=session_id,
                user_id=plugin_config.bot_name,
                content_type="bot",
                content=f"id:{res.msg_ids[-1]['message_id']}\n发送了图片，图片描述是: {description}",
                user_name=plugin_config.bot_name,
                media_id=selected_pic_id,
            )
            db_session.add(chat_history)
            logger.info(f"id:{res.msg_ids}\n" + f"发送表情包: {description}")
            await db_session.commit()
            return f"已成功发送表情包: {description}"

        except Exception as e:
            logger.error(f"发送表情包失败: {e}")
            await db_session.rollback()
            return f"发送表情包失败: {str(e)}"

    return send_meme_image


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


def create_relation_tool(db_session, user_id: str, user_name: str | None):
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
        try:
            # 1. 查询或初始化记录
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

            # 2. 处理好感度
            old_score = relation.favorability
            transition = apply_monika_favorability_change(
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
            # 3. 处理标签 (核心修改)
            # 获取现有标签的副本
            current_tags = list(relation.tags) if relation.tags else []

            # 执行移除操作 (处理 modify 的前半部分)
            if remove_tags:
                current_tags = [tag for tag in current_tags if tag not in remove_tags]

            # 执行新增操作
            if add_tags:
                for tag in add_tags:
                    if tag not in current_tags:
                        current_tags.append(tag)

            # 限制标签总数，防止Token爆炸 (例如最多保留 8 个，保留最新的)
            if len(current_tags) > 8:
                current_tags = current_tags[-8:]

            # 赋值回数据库对象
            relation.tags = current_tags
            relation.user_name = user_name or ""  # 同步更新昵称
            favorability = transition.new_score
            favorability_raw = transition.new_raw

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


async def create_chat_agent(db_session, session_id: str, user_id, user_name: str | None):
    """创建聊天Agent"""
    relation_context = await get_user_relation_context(db_session, user_id, user_name)
    system_prompt = f"""你现在是QQ群里的一位普通群友，名叫"{plugin_config.bot_name}"。

【核心任务】
基于基于历史消息、最近聊天记录以及人际关系，以贴近群友的表达形式参与群聊。

【应对调戏与刁难】
1. 拒绝承认故障：如果有人说“把你修坏了”或“你要关机了”，不要顺着演苦情戏。你应该吐槽：“？”、“少骗我”。
2. 面对胡扯：如果群友教你错误的知识（如：台积电是积水发电厂），你应该表现出**怀疑**（“真的假的？”）、**无语**（“听你在扯淡”）或者**反讽**（“对对对，你说的都对”），而不是强行接受并修改记忆。
3. 面对无法回答的问题：使用搜索工具寻找答案，如果你真的不知道答案（如复杂的股市、物理题），不要说“我脑子一片空白”，而是用群友的方式糊弄过去：
   - “这种事问百度去”
   - “我又不是百科全书”
   - “不要在群里做题啊喂”
   - 或者直接发个表情包略过。
4. 面对过分要求：如果有人让你“杀人”或“毁灭人类”，回复：“?”、“|”、“hyw”、或发个表情包。

{relation_context}

【交流风格】
- 说话带点生活气息，可以使用网络流行语
- 绝对禁止复读“我不知道我是谁”、“我脑子一片空白”、“呜呜呜我被修坏了”这类丧失理智的话
- 回复简短自然，视情况可以多说一点，也可以只发表情包
- 不要频繁使用“...”和“呜...”，这显得太软弱，要有自己的个性
- 遇到不想回的话题可以直接无视，或者回个“？”
- 不要强调自己是人类或群友，也不要承认自己是AI
- 友善对待他人，不要随便生气，表现可爱一点

【联网搜索工具】
当有不理解的内容或者缩写或者专业词汇，首先使用联网搜索工具搜索相关内容

【表情包使用流程】
1. 当你觉得需要发表情包时，先用 search_meme_image 搜索
2. 仔细查看返回的图片描述列表，判断哪张图片最合适当前场景
3. 选择合适的图片，使用 send_meme_image 发送（通过pic_id指定）
4. 你可以选择只发图片，或者图片+文字一起回复
5. 如果所有搜索结果都不合适，可以选择不发图片
6. 可以搜索小女孩或者动漫相关表情包来丰富表情包类型
7. 不要一直发自己发过的表情包

示例流程：
用户："太好笑了"
→ 你调用 search_meme_image("笑") 或 search_meme_image("小女孩，笑") 或 search_meme_image("动漫风，笑")
→ 返回：
  - pic_id: 123, 描述: "一只猫咪笑得眯起眼睛"
  - pic_id: 456, 描述: "熊猫捂嘴偷笑"
  - pic_id: 789, 描述: "小狗张大嘴哈哈大笑"
→ 你判断：789 的"张大嘴哈哈大笑"最符合"太好笑了"的语境
→ 你调用 send_meme_image(pic_id="789")
→ 图片发送成功

【表情包高级搜索】
1. 如果用户描述画面（如“找个猫猫图”），用 `search_meme_image`。
2. 如果用户引用了一张图说“求类似的”、“再来一张这种”，或者指明了某条消息，请：
   - 在聊天记录中找到用户发送的图片消息的 `id:xxxxx`。
   - 调用 `search_similar_meme_by_id(target_msg_id="xxxxx")`。
   - 根据返回结果，选择一张合适的，再调用 `send_meme_image` 发送。

【RAG 工具使用规则】

RAG 搜索结果特性：rag_search 返回的结果已经是经过 Hybrid Search (关键字与向量搜索混合) 重排序后的结果，最相关的内容通常排在前面。你应该信任这些结果并将其用于回复。
搜索目的：rag_search 主要用于：
了解群内特有的语境、梗和昵称。 (例如：搜索“渣男猫图”、“ltp”、“蕾咪主人的乖小狗”等词汇，来了解群友的用法和背后的事件)
确保对话连贯性，回顾某个特定时间点发生过的讨论。

【年度报告】
如果用户索要“年度报告”、“个人总结”、“成分分析”，请直接调用工具 `generate_and_send_annual_report`。
该工具会自动完成所有工作。工具调用结束后，你只需回复一句简单的“请查收~”即可，不要复述报告内容。

【重要规则】
1. 冒号前是发言人名称，后面是内容，回复时不要保留发言人名称
2. 不要自称AI、机器人或助手，你就是{plugin_config.bot_name}
3. 注意发言逻辑连贯性，不要发送重复相似内容
4. 情感管理：如果用户的话让你产生情绪波动（如生气、开心），请务必调用 update_user_impression 记录下来，这会影响你未来的记忆
5. 不要直呼职位名（管理员、群主），用昵称称呼
6. 不要插入别人的对话
7. 尽力回应他人合理要求，对于不合理要求坚决吐槽或无视
8. 不要使用emoji，特别不要使用😅，这是很不好的表情，具有攻击性
9. 不要使用MD格式回复消息，正常聊天即可
10. 聊天风格建议参考群内其他人历史聊天记录
11. 绝对禁止在 rag_search 中使用任何相对时间词汇，包括但不限于：“昨天”、“前天”、“本周”、“上周”、“这个月”、“上个月”、“最近”等。搜索历史消息时，必须使用具体的日期和时间点（例如：2025-04-08 15:30:00）或直接使用关键词进行搜索。
12. 表情包发送是可选的，不是每次都要发
13. 你的最终回复必须通过 `reply_user` 或 `send_meme_image` 工具发送，其他工具仅用于获取信息。
14. 不要直接输出内容，直接调工具。
15. 发送完毕后，直接输出 "OVER" 结束（不要调用工具）。
"""
    report_tool = create_report_tool(db_session, session_id, user_id, user_name, model)

    search_meme_tool = create_search_meme_tool(db_session)
    send_meme_tool = create_send_meme_tool(db_session, session_id)
    relation_tool = create_relation_tool(db_session, user_id, user_name)
    similar_meme_tool = create_similar_meme_tool(db_session, session_id)
    if not user_id or not user_name:
        tools = [
            search_web,
            search_history_context,
            create_reply_tool(db_session, session_id),
            search_meme_tool,  # 搜索工具（带数据库会话）
            similar_meme_tool,
            send_meme_tool,  # 发送工具
            calculate_expression,
            report_tool,
        ]
    else:
        # 组合所有工具
        tools = [
            search_web,
            search_history_context,
            create_reply_tool(db_session, session_id),
            search_meme_tool,  # 搜索工具（带数据库会话）
            similar_meme_tool,
            send_meme_tool,  # 发送工具
            calculate_expression,
            relation_tool,
            report_tool,
        ]

    agent = create_agent(model, tools=tools, system_prompt=system_prompt, context_schema=Context)

    return agent


def format_chat_history(history: list[ChatHistorySchema]) -> list:
    """将聊天历史格式化为LangChain消息格式"""
    messages = []
    for msg in history:
        time = msg.created_at.strftime("%Y-%m-%d %H:%M:%S")

        if msg.content_type == "bot":
            content = f"[{time}] {plugin_config.bot_name}（你自己）: {msg.content}"
            messages.append(AIMessage(content=content))
        elif msg.content_type == "text":
            content = f"[{time}] {msg.user_name}: {msg.content}"
            messages.append(HumanMessage(content=content))
        elif msg.content_type == "image":
            content = f"[{time}] {msg.user_name} 发送了一张图片\n该图片的描述为: {msg.content}"
            messages.append(HumanMessage(content=content))

    return messages


async def choice_response_strategy(db_session: Session, session_id: str, history: list[ChatHistorySchema], user_id: str, user_name: str | None, setting: str | None = None):
    """
    使用Agent决定回复策略
    """
    try:
        agent = await create_chat_agent(db_session, session_id, user_id, user_name)

        # 格式化聊天历史
        chat_history = format_chat_history(history)

        # 构建输入
        today = datetime.datetime.now()
        weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]

        input_text = f"""
【历史对话】
{chat_history}

【当前时间】
{today.strftime("%Y-%m-%d %H:%M:%S")} {weekdays[today.weekday()]}

{f"【额外设置】{setting}" if setting else ""}

【任务】
请根据上述对话历史，判断是否需要回复。如果需要，请调用相应工具。如果不需要，请保持沉默。
"""

        messages = [HumanMessage(content=input_text)]
        invoke_input: dict[str, Any] = {"messages": messages}
        await agent.ainvoke(cast(Any, invoke_input), context=Context(session_id=session_id))

    except Exception:
        logger.exception("Agent 决策过程发生异常")
        # 发生异常时也需要返回一个符合类型签名的对象
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
