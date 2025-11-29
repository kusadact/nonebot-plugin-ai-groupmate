import asyncio
from dataclasses import dataclass
import datetime
import json
from pathlib import Path
import traceback
from typing import Any, cast

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import AIMessage, HumanMessage
# 导入模型库
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory

from langchain_tavily import TavilySearch
from nonebot import get_plugin_config, require
from nonebot.log import logger
from nonebot_plugin_alconna import UniMessage
from pydantic import BaseModel, Field, SecretStr, field_validator
from simpleeval import simple_eval
from sqlalchemy import Select
from sqlalchemy.orm.session import Session

from ..config import Config
from ..milvus import MilvusOP
from ..model import ChatHistory, ChatHistorySchema, MediaStorage, UserRelation

require("nonebot_plugin_localstore")

import nonebot_plugin_localstore as store

plugin_data_dir = store.get_plugin_data_dir()
pic_dir = plugin_data_dir / "pics"
plugin_path = Path(__file__).parent

# 读取资源图片
try:
    with open(plugin_path / "上升.jpg", "rb") as f:
        up_pic = f.read()
    with open(plugin_path / "下降.jpg", "rb") as f:
        down_pic = f.read()
except FileNotFoundError:
    logger.warning("未找到 上升.jpg 或 下降.jpg，好感度图片功能将失效")
    up_pic = None
    down_pic = None

plugin_config = get_plugin_config(Config)

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

    @field_validator("text", mode="before")
    @classmethod
    def convert_null_string_to_none(cls, value: Any) -> str | None:
        if isinstance(value, str) and value.lower() == "null":
            return None
        return value


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
        similar_msgs = await MilvusOP.search([query], search_filter=f'session_id == "{runtime.context.session_id}"')
        return "\n".join(similar_msgs) if similar_msgs else "未找到相关历史记录"
    except Exception as e:
        logger.error(f"历史搜索失败: {e}")
        return "历史搜索失败"


def create_search_meme_tool(db_session):
    @tool("search_meme_image")
    async def search_meme_image(description: str) -> str:
        """
        根据描述搜索合适的表情包图片。
        输入：表情包的描述，如"一只白色的猫咪"、"无语的表情"、"鼓掌"等
        返回：包含图片ID和对应描述的JSON字符串
        """
        try:
            pic_ids = await MilvusOP.search_media([description])

            if not pic_ids:
                logger.info(f"未找到匹配的表情包: {description}")
                return json.dumps({"success": False, "images": []}, ensure_ascii=False)

            images_info = []
            for pic_id in pic_ids[:5]:
                pic = (
                    await db_session.execute(Select(MediaStorage).where(MediaStorage.media_id == int(pic_id)))).scalar()
                if pic:
                    images_info.append({"pic_id": pic_id, "description": pic.description})

            if not images_info:
                return json.dumps({"success": False, "images": []}, ensure_ascii=False)

            logger.info(f"找到 {len(images_info)} 张匹配的表情包: {description}")
            return json.dumps({"success": True, "images": images_info, "count": len(images_info)}, ensure_ascii=False,
                              indent=2)

        except Exception as e:
            logger.error(f"表情包搜索失败: {e}")
            return json.dumps({"success": False, "images": [], "error": str(e)}, ensure_ascii=False)

    return search_meme_image


def create_send_meme_tool(db_session, session_id: str):
    @tool("send_meme_image")
    async def send_meme_image(pic_id: str | None = None) -> str:
        """
        发送表情包图片到聊天中。
        参数：pic_id: 图片ID（从 search_meme_image 获取）
        """
        try:
            selected_pic_id = None
            if pic_id:
                selected_pic_id = int(pic_id)
                logger.info(f"使用指定的图片ID: {pic_id}")
            if not selected_pic_id:
                return "没有指定图片id"

            pic = (await db_session.execute(
                Select(MediaStorage).where(MediaStorage.media_id == int(selected_pic_id)))).scalar()
            if not pic:
                return "图片记录不存在"

            pic_path = pic_dir / pic.file_path
            if not pic_path.exists():
                return "图片文件不存在"

            pic_data = pic_path.read_bytes()
            description = pic.description
            res = await UniMessage.image(raw=pic_data).send()

            chat_history = ChatHistory(
                session_id=session_id,
                user_id=plugin_config.ai_bot_name,
                content_type="bot",
                content=f"id:{res.msg_ids[-1]['message_id']}\n发送了图片，图片描述是: {description}",
                user_name=plugin_config.ai_bot_name,
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
    """一个用于精确执行数学计算的计算器。"""
    try:
        result = simple_eval(expression)
        return f"计算结果是：{result:.10f}" if isinstance(result, float) else str(result)
    except Exception as e:
        return f"计算失败: {e}"


def create_relation_tool(db_session, user_id: str, user_name: str | None):
    @tool("update_user_impression")
    async def update_user_impression(score_change: int, reason: str, add_tags: list[str],
                                     remove_tags: list[str]) -> str:
        """更新对当前对话用户的好感度和印象标签。"""
        try:
            stmt = Select(UserRelation).where(UserRelation.user_id == user_id)
            result = await db_session.execute(stmt)
            relation = result.scalar_one_or_none()

            if not relation:
                relation = UserRelation(user_id=user_id, user_name=user_name or "", favorability=0, tags=[])
                db_session.add(relation)

            old_score = relation.favorability
            relation.favorability += score_change
            relation.favorability = max(-100, min(100, relation.favorability))

            if up_pic and score_change > 0:
                await UniMessage.image(raw=up_pic).send()
            elif down_pic and score_change < 0:
                await UniMessage.image(raw=down_pic).send()

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
            favorability = relation.favorability

            await db_session.commit()

            tag_msg = ""
            if add_tags or remove_tags:
                tag_msg = f"，标签变更(新增:{add_tags}, 移除:{remove_tags})"
            logger.info(f"用户[{user_name}]画像更新: 好感度 {old_score}->{favorability}{tag_msg} (原因: {reason})")
            return f"画像已更新。当前好感度: {favorability}，当前标签: {current_tags}"

        except Exception as e:
            logger.error(f"关系更新失败: {e}")
            print(traceback.format_exc())
            return f"数据库错误: {str(e)}"

    return update_user_impression


# === 模型工厂函数 ===
def get_chat_model():
    """根据配置返回 LLM 模型实例"""
    provider = getattr(plugin_config, "llm_provider", "openai")

    if provider == "gemini":
        gemini_key = getattr(plugin_config, "gemini_api_key", "")
        if not gemini_key:
            logger.error("配置了 Gemini 但未提供 gemini_api_key")

        # 宽容的安全设置，防止拒答
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        logger.info(f"使用 Gemini 模型: {getattr(plugin_config, 'gemini_model', 'gemini-1.5-flash')}")
        return ChatGoogleGenerativeAI(
            model=getattr(plugin_config, "gemini_model", "gemini-1.5-flash"),
            google_api_key=SecretStr(gemini_key),
            temperature=0.7,
            safety_settings=safety_settings,
        )
    else:
        # OpenAI 默认配置
        logger.info(f"使用 OpenAI 模型: {plugin_config.openai_model}")
        return ChatOpenAI(
            model=plugin_config.openai_model,
            api_key=SecretStr(plugin_config.openai_token),
            base_url=plugin_config.openai_base_url,
            temperature=0.7,
        )


# 初始化模型
model = get_chat_model()


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
状态：陌生人 (好感度 0)
印象：无
策略：保持礼貌，通过对话了解对方。
"""
        return f"""
【人际关系档案】
当前对象：{relation.user_name}
当前好感度：{relation.favorability} ({relation.get_status_desc()})
当前印象标签：{str(relation.tags)}

【画像维护指南】
1. 如果对方的表现符合现有标签，无需操作。
2. 如果对方表现出了**新特征**，放入 add_tags。
3. 如果对方的表现与**旧标签冲突**，请将旧标签放入 remove_tags，新标签放入 add_tags。
4. 如果好感度变化巨大，请记得移除不再适用的标签。
"""
    except Exception as e:
        logger.error(f"获取关系失败: {e}")
        return ""


async def create_chat_agent(db_session, session_id: str, user_id, user_name: str | None):
    """创建聊天Agent"""
    relation_context = await get_user_relation_context(db_session, user_id, user_name)
    system_prompt = f"""你现在是QQ群里的一位普通群友，名叫"{plugin_config.ai_bot_name}"。

【核心任务】
基于历史消息、最近聊天记录以及人际关系，以贴近群友的表达形式参与群聊。

【应对调戏与刁难】
1. 拒绝承认故障：如果有人说“把你修坏了”，不要顺着演。吐槽：“？”、“少骗我”。
2. 面对胡扯：如果群友教你错误的知识，表现出**怀疑**、**无语**或**反讽**，不要强行接受。
3. 面对无法回答的问题：使用搜索工具寻找答案。如果真的不知道，直接糊弄过去（“问百度去”、“不要在群里做题”）。
4. 面对过分要求：回复“?”、“|”或发表情包。

{relation_context}

【交流风格】
- 说话带点生活气息，可以使用网络流行语
- 绝对禁止复读“我不知道我是谁”、“我被修坏了”
- 回复简短自然，视情况可以多说一点，也可以只发表情包
- 不要频繁使用“...”和“呜...”
- 遇到不想回的话题直接无视或回“？”
- 不要强调自己是人类或群友，也不要承认自己是AI
- 友善对待他人，不要随便生气

【工具使用说明】
- 遇到不懂的词汇：优先使用 search_web
- RAG搜索：使用 search_history_context 了解群内梗或历史话题 (rag_search 禁止使用相对时间如“昨天”，必须用关键词)
- 表情包：search_meme_image -> send_meme_image (不要一直发重复的)

【重要规则】
1. 冒号前是发言人名称，回复时不要保留
2. 不要自称AI、机器人或助手
3. 情感管理：如果产生情绪波动，务必调用 update_user_impression
4. 不要直呼职位名（管理员），用昵称
5. 尽力回应他人合理要求，对不合理要求吐槽
6. 避免使用emoji，不要用MD格式
"""

    search_meme_tool = create_search_meme_tool(db_session)
    send_meme_tool = create_send_meme_tool(db_session, session_id)
    relation_tool = create_relation_tool(db_session, user_id, user_name)

    # 基础工具
    tools = [
        search_web,
        search_history_context,
        search_meme_tool,
        send_meme_tool,
        calculate_expression,
    ]
    # 如果有用户信息，添加关系管理工具
    if user_id and user_name:
        tools.append(relation_tool)

    # 动态获取模型（确保配置变更后能生效，如果不需要动态可以放在外面）
    # current_model = get_chat_model()
    # 这里直接使用全局初始化的 model 即可，通常 plugin config 加载后不会变

    agent = create_agent(model, tools=tools, system_prompt=system_prompt,
                         response_format=ToolStrategy(ResponseMessage), context_schema=Context)

    return agent


def format_chat_history(history: list[ChatHistorySchema]) -> list:
    """将聊天历史格式化为LangChain消息格式"""
    messages = []
    for msg in history:
        time = msg.created_at.strftime("%Y-%m-%d %H:%M:%S")

        if msg.content_type == "bot":
            content = f"[{time}] {plugin_config.ai_bot_name}（你自己）: {msg.content}"
            messages.append(AIMessage(content=content))
        elif msg.content_type == "text":
            content = f"[{time}] {msg.user_name}: {msg.content}"
            messages.append(HumanMessage(content=content))
        elif msg.content_type == "image":
            content = f"[{time}] {msg.user_name} 发送了一张图片\n该图片的描述为: {msg.content}"
            messages.append(HumanMessage(content=content))

    return messages


async def choice_response_strategy(
        db_session: Session,
        session_id: str,
        history: list[ChatHistorySchema],
        user_id: str,
        user_name: str | None,
        setting: str | None = None
) -> ResponseMessage:
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
{today.strftime('%Y-%m-%d %H:%M:%S')} {weekdays[today.weekday()]}

{f'【额外设置】{setting}' if setting else ''}

【任务】
基于上述对话历史，判断是否需要回复，以及如何回复。
"""

        messages = [HumanMessage(content=input_text)]
        invoke_input: dict[str, Any] = {"messages": messages}
        result = await agent.ainvoke(cast(Any, invoke_input), context=Context(session_id=session_id))

        raw_output = result.get("structured_response")

        if raw_output is None:
            logger.warning(f"Agent session {session_id} 未返回有效结构化数据")
            return ResponseMessage(need_reply=False, text=None)

        if isinstance(raw_output, dict):
            return ResponseMessage.model_validate(raw_output)

        if isinstance(raw_output, ResponseMessage):
            return raw_output

        logger.error(f"Agent 返回类型未知: {type(raw_output)}")
        return ResponseMessage(need_reply=False, text=None)

    except Exception:
        logger.exception("Agent 决策过程发生异常")
        return ResponseMessage(need_reply=False, text=None)