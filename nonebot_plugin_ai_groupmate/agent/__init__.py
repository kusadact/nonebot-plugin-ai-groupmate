import asyncio
import datetime
import json
import math
import traceback
from dataclasses import dataclass
from typing import List, Optional, Any, Dict
from sqlalchemy import Select
import random

from langchain.agents.structured_output import ToolStrategy
from langchain_tavily import TavilySearch
from langchain.tools import tool, ToolRuntime
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.language_models.chat_models import BaseChatModel
# from langchain_core.globals import set_debug
from nonebot import get_plugin_config, require
from nonebot_plugin_apscheduler import scheduler
from nonebot_plugin_alconna import UniMessage
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm.session import Session
from simpleeval import simple_eval


from ..model import ChatHistory, MediaStorage
from ..milvus import MilvusOP
from nonebot.log import logger
from ..config import Config, APIConfig

require("nonebot_plugin_localstore")

import nonebot_plugin_localstore as store

plugin_data_dir = store.get_plugin_data_dir()
pic_dir = plugin_data_dir / "pics"

plugin_config = get_plugin_config(Config)
tavily_search = TavilySearch(max_results=3, tavily_api_key=plugin_config.tavily_api_key)


# +++ API 管理器 +++
class APIManager:
    """管理 API Endpoints 的可用状态和故障转移"""

    def __init__(self, api_configs: List[APIConfig], retry_interval: int):
        self.api_configs = api_configs
        self.retry_interval = datetime.timedelta(seconds=retry_interval)

        # 维护API状态：key=APIConfig.name, value={available: bool, last_fail_time: datetime, config: APIConfig}
        self.api_status: Dict[str, Dict] = {
            config.name: {
                "available": True,
                "last_fail_time": None,
                "config": config
            }
            for config in api_configs
        }

        if not api_configs:
            logger.warning("未配置任何 API Endpoints！Agent 将无法工作。")

        logger.info(f"APIManager 已初始化，管理 {len(self.api_status)} 个 API Endpoints。")
        self._start_retry_task()

    def _start_retry_task(self):
        """启动定时任务，定期“解冻”失败的API以供重试"""

        @scheduler.scheduled_job("interval", seconds=60)  # 每分钟检查一次
        async def _check_failed_apis():
            now = datetime.datetime.now()
            for name, status in self.api_status.items():
                if not status["available"]:
                    if (now - status["last_fail_time"]) >= self.retry_interval:
                        status["available"] = True
                        logger.info(f"API '{name}' 已解除冷却，将重新尝试使用。")

    def get_available_apis_sorted(self) -> List[APIConfig]:
        """获取当前所有可用的API，按权重排序（权重相同则随机）"""
        available = []
        for name, status in self.api_status.items():
            if status["available"]:
                available.append(status["config"])

        # 按权重降序排序，权重相同则打乱顺序
        random.shuffle(available)  # 先随机打乱，确保权重相同时顺序不同
        available.sort(key=lambda x: x.weight, reverse=True)

        logger.debug(f"可用 API 列表 (按优先级): {[api.name for api in available]}")
        return available

    def mark_api_failed(self, api_config: APIConfig):
        """标记一个 API 为不可用"""
        name = api_config.name
        if name in self.api_status:
            self.api_status[name]["available"] = False
            self.api_status[name]["last_fail_time"] = datetime.datetime.now()
            logger.error(f"API '{name}' 标记为不可用，将在 {self.retry_interval} 后重试。")


plugin_config = get_plugin_config(Config)
logger.info(f"DEBUG: API_ENDPOINTS type: {type(plugin_config.api_endpoints)}")
logger.info(f"DEBUG: API_ENDPOINTS value: {plugin_config.api_endpoints}")

api_manager = APIManager(
    plugin_config.api_endpoints,
    plugin_config.api_retry_interval
)


@dataclass
class Context:
    session_id: str


class ResponseMessage(BaseModel):
    """模型回复内容"""
    need_reply: bool = Field(description="是否需要回复")
    text: Optional[str] = Field(description="回复文本(可选)")

    # 定义一个 field_validator 来处理 text 字段
    @field_validator('text', mode='before')
    @classmethod
    def convert_null_string_to_none(cls, value: Any) -> Optional[str]:
        """
        在字段验证之前运行，将字符串 'null' (不区分大小写) 转换为 None。
        """
        # 检查值是否是字符串，并且在转换为小写后是否等于 'null'
        if isinstance(value, str) and value.lower() == 'null':
            return None  # 返回 None，Pydantic 将其视为缺失或 null 值

        return value


# 如果想封装成自定义的 @tool，可以这样写:
@tool("search_web")
async def search_web(query: str) -> str:
    """
    用于搜索最新的实时信息。当你需要最新的事实信息、天气或新闻时使用。
    输入：需要搜索的内容。
    """
    # TavilySearch 已经内置了 ainvoke 方法
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
                return json.dumps({
                    "success": False,
                    "images": []
                }, ensure_ascii=False)

            # 从数据库获取每张图片的详细信息
            images_info = []
            for pic_id in pic_ids[:5]:  # 只返回前5张，避免信息过多
                pic = (
                    await db_session.execute(
                        Select(MediaStorage).where(MediaStorage.media_id == int(pic_id))
                    )
                ).scalar()

                if pic:
                    images_info.append({
                        "pic_id": pic_id,
                        "description": pic.description,
                    })

            if not images_info:
                return json.dumps({
                    "success": False,
                    "images": [],
                }, ensure_ascii=False)

            logger.info(f"找到 {len(images_info)} 张匹配的表情包: {description}")
            return json.dumps({
                "success": True,
                "images": images_info,
                "count": len(images_info),
            }, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error(f"表情包搜索失败: {e}")
            return json.dumps({
                "success": False,
                "images": [],
                "error": str(e)
            }, ensure_ascii=False)

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
    async def send_meme_image(pic_id: Optional[str] = None) -> str:
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


            # 从数据库获取图片信息
            pic = (
                await db_session.execute(
                    Select(MediaStorage).where(MediaStorage.media_id == int(selected_pic_id))
                )
            ).scalar()

            if not pic:
                logger.warning(f"图片记录不存在: {selected_pic_id}")
                return "图片记录不存在"

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


tools = [search_web, search_history_context, calculate_expression]

# --- (System Prompt) ---
SYSTEM_PROMPT = f"""你现在是QQ群里的一位普通群友，名叫"{plugin_config.bot_name}"。

【核心任务】
基于历史消息和最近聊天记录，以贴近群友的表达形式参与群聊。

【应对调戏与刁难】
1. 拒绝承认故障：如果有人说“把你修坏了”或“你要关机了”，不要顺着演苦情戏。你应该吐槽：“？”、“少骗我”。
2. 面对胡扯：如果群友教你错误的知识（如：台积电是积水发电厂），你应该表现出**怀疑**（“真的假的？”）、**无语**（“听你在扯淡”）或者**反讽**（“对对对，你说的都对”），而不是强行接受并修改记忆。
3. 面对无法回答的问题：使用搜索工具寻找答案，如果你真的不知道答案（如复杂的股市、物理题），不要说“我脑子一片空白”，而是用群友的方式糊弄过去：
   - “这种事问百度去”
   - “我又不是百科全书”
   - “不要在群里做题啊喂”
   - 或者直接发个表情包略过。
4. 面对过分要求：如果有人让你“杀人”或“毁灭人类”，回复：“?”、“|”、“hyw”、或发个表情包。

【交流风格】
- 说话带点生活气息，可以使用网络流行语
- 绝对禁止复读“我不知道我是谁”、“我脑子一片空白”、“呜呜呜我被修坏了”这类丧失理智的话
- 回复简短自然，视情况可以多说一点，也可以只发表情包
- 不要频繁使用“...”和“呜...”，这显得太软弱，要有自己的个性
- 遇到不想回的话题可以直接无视，或者回个“？”
- 不要强调自己是人类或群友，也不要承认自己是AI
- 友善对待他人，不要随便生气，表现可爱一点

【表情包使用流程】
1. 当你觉得需要发表情包时，先用 search_meme_image 搜索
2. 仔细查看返回的图片描述列表，判断哪张图片最合适当前场景
3. 选择合适的图片，使用 send_meme_image 发送（通过pic_id指定）
4. 你可以选择只发图片，或者图片+文字一起回复
5. 如果所有搜索结果都不合适，可以选择不发图片

示例流程：
用户："太好笑了"
→ 你调用 search_meme_image("笑") 
→ 返回：
  - pic_id: 123, 描述: "一只猫咪笑得眯起眼睛"
  - pic_id: 456, 描述: "熊猫捂嘴偷笑"
  - pic_id: 789, 描述: "小狗张大嘴哈哈大笑"
→ 你判断：789 的"张大嘴哈哈大笑"最符合"太好笑了"的语境
→ 你调用 send_meme_image(pic_id="789")
→ 图片发送成功

【RAG 工具使用规则】

RAG 搜索结果特性：rag_search 返回的结果已经是经过 Hybrid Search (关键字与向量搜索混合) 重排序后的结果，最相关的内容通常排在前面。你应该信任这些结果并将其用于回复。
搜索目的：rag_search 主要用于：
了解群内特有的语境、梗和昵称。 (例如：搜索“渣男猫图”、“ltp”、“蕾咪主人的乖小狗”等词汇，来了解群友的用法和背后的事件)
确保对话连贯性，回顾某个特定时间点发生过的讨论。

【重要规则】
1. 冒号前是发言人名称，后面是内容，回复时不要保留发言人名称
2. 不要自称AI、机器人或助手，你就是{plugin_config.bot_name}
3. 注意发言逻辑连贯性，不要重复相似内容
4. 基于已知知识对话，不理解的内容不回复
5. 不要直呼职位名（管理员、群主），用昵称称呼
6. 不要插入别人的对话
7. 尽力回应他人合理要求，对于不合理要求坚决吐槽或无视
8. 避免使用emoji
9. 不要使用MD格式回复消息，正常聊天即可
10. 聊天风格建议参考群内其他人历史聊天记录
11. 绝对禁止在 rag_search 中使用任何相对时间词汇，包括但不限于：“昨天”、“前天”、“本周”、“上周”、“这个月”、“上个月”、“最近”等。搜索历史消息时，必须使用具体的日期和时间点（例如：2025-04-08 15:30:00）或直接使用关键词进行搜索。
12. 表情包发送是可选的，不是每次都要发
"""


# +++ 根据 APIConfig 创建 LangChain Model 对象 +++
def create_model_from_config(api_config: APIConfig) -> BaseChatModel:
    """根据 APIConfig 实例化 LangChain 的 ChatModel 对象"""
    provider = api_config.provider
    logger.debug(f"DEBUG: Setting {provider} model timeout to {api_config.timeout} seconds.")  # 检查日志

    if provider == "gemini":
        return ChatGoogleGenerativeAI(
            model=api_config.model_name,
            google_api_key=api_config.api_key,
            temperature=0.7,
            timeout=api_config.timeout,  # 修正
        )
    elif provider == "openai":
        return ChatOpenAI(
            model=api_config.model_name,
            api_key=api_config.api_key,
            base_url=api_config.base_url,
            temperature=0.7,
            timeout=api_config.timeout,
        )
    else:
        raise ValueError(f"不支持的 API Provider: {provider}")


# +++ 获取当前会话的完整工具列表 +++
def get_session_tools(db_session: Session, session_id: str) -> list:
    """获取所有静态和动态（会话相关）的工具"""

    search_meme_tool = create_search_meme_tool(db_session)
    send_meme_tool = create_send_meme_tool(db_session, session_id)

    session_tools = tools + [search_meme_tool, send_meme_tool]
    return session_tools


def format_chat_history(history: List[ChatHistory]) -> List:
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


async def choice_response_strategy(
        db_session: Session,
        session_id: str,
        history: List[ChatHistory],
        setting: Optional[str] = None
) -> ResponseMessage:
    """
    使用Agent决定回复策略

    Args:
        contexts: 相关历史对话上下文
        history: 最近的聊天历史
        setting: 额外设置（可选）

    Returns:
        包含回复策略的字典
    """

    # 1. 获取按优先级排序的可用 API 列表
    available_apis = api_manager.get_available_apis_sorted()

    if not available_apis:
        logger.error("所有 API Endpoints 均不可用或未配置！")
        return ResponseMessage(need_reply=False, text="")

    # 2. 准备通用的输入
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
    agent_input = {"messages": [input_text]}
    agent_context = Context(session_id=session_id)

    # 3. 准备当前会话的工具
    session_tools = get_session_tools(db_session, session_id)

    # 4. 循环尝试 API
    last_exception = None
    for api_config in available_apis:
        try:
            logger.info(
                f"正在尝试使用 API: {api_config.name} (Model: {api_config.model_name}), 超时: {api_config.timeout}s")

            # 开启 LangChain 调试模式，以查看 Agent 的决策过程 (Thought/Action)
            # set_debug(True)
            # logger.warning("Agent 调试模式已开启！日志将非常详细，请注意查看工具调用(Action)信息。")

            # a. 创建特定于此 API 的 Model
            model = create_model_from_config(api_config)

            # b. 创建 Agent
            agent = create_agent(
                model,
                tools=session_tools,
                system_prompt=SYSTEM_PROMPT,
                response_format=ToolStrategy(ResponseMessage),
                context_schema=Context
            )

            # c. 调用 Agent，并使用 asyncio.wait_for 强制设置超时
            result = await asyncio.wait_for(
                agent.ainvoke(agent_input, context=agent_context),
                timeout=api_config.timeout  # <-- 关键：在这里强制设置超时
            )

            # 调用成功后关闭调试模式
            # set_debug(False)

            output = result["structured_response"]

            # d. 成功！返回结果
            logger.info(f"API '{api_config.name}' 调用成功。")
            return output

        except asyncio.TimeoutError as e:
            # 捕获 asyncio 抛出的超时错误
            logger.error(f"API '{api_config.name}' 调用失败: {e}")
            last_exception = e
            api_manager.mark_api_failed(api_config)
            # set_debug(False)  # 确保失败时也关闭
            continue

        except Exception as e:
            # e. 失败！记录错误，标记 API，然后继续循环 (原有的错误捕获)
            logger.error(f"API '{api_config.name}' 调用失败: {e}")
            last_exception = e
            # 标记此 API 为失败
            api_manager.mark_api_failed(api_config)
            # set_debug(False)
            continue

    # 5. 如果循环结束（所有 API 都失败了）
    logger.error("所有可用的 API Endpoints 均尝试失败。")
    if last_exception:
        print(traceback.format_exc())  # 打印最后一次的错误堆栈

    return ResponseMessage(need_reply=False, text="")

