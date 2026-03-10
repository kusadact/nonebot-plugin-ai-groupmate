import asyncio
import datetime
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Literal

from langchain_openai import ChatOpenAI
from nonebot import get_plugin_config
from nonebot.log import logger
from nonebot_plugin_orm import AsyncSession
from pydantic import BaseModel, Field, SecretStr
from sqlalchemy import Select, Update

from .config import Config
from .mem0_client import mem0_client
from .model import ChatHistory, ChatHistorySchema
from .utils import estimate_token_count

plugin_config = get_plugin_config(Config).ai_groupmate

PROFILE_MEMORY_TYPES = (
    "speaking_style",
    "interests",
    "dislikes",
    "relationship_tone",
    "recent_topics",
    "stable_traits",
)
MAX_WINDOW_TIME_GAP = timedelta(minutes=10)
MAX_WINDOW_MESSAGES = 120
MAX_WINDOW_TOKENS = 5000
MIN_USER_MESSAGES_PER_WINDOW = 30
MAX_FACTS_PER_WINDOW_USER = 6
MAX_FACTS_PER_TYPE = 2
MAX_FINAL_FACTS_PER_USER = 8
MIN_RECENT_TOPIC_CONFIDENCE = 0.60
MIN_STABLE_FACT_CONFIDENCE = 0.72
HIGH_CONFIDENCE_DIRECT_WRITE = 0.86


class ProfileFactCandidate(BaseModel):
    memory_type: Literal[
        "speaking_style",
        "interests",
        "dislikes",
        "relationship_tone",
        "recent_topics",
        "stable_traits",
    ] = Field(description="画像类型")
    content: str = Field(description="画像内容，必须是简短中文短语，不要写完整句子。")
    confidence: float = Field(ge=0.0, le=1.0, description="置信度，范围 0 到 1。")
    evidence_msg_ids: list[int] = Field(default_factory=list, description="证据消息 ID。")


class ProfileExtractionResult(BaseModel):
    facts: list[ProfileFactCandidate] = Field(default_factory=list)


@dataclass
class AggregatedProfileFact:
    memory_type: str
    content: str
    user_name: str
    confidence_sum: float = 0.0
    confidence_max: float = 0.0
    occurrences: int = 0
    evidence_msg_ids: set[int] = field(default_factory=set)
    last_seen_at: datetime.datetime | None = None

    def add(self, fact: ProfileFactCandidate, user_name: str, seen_at: datetime.datetime) -> None:
        self.confidence_sum += fact.confidence
        self.confidence_max = max(self.confidence_max, fact.confidence)
        self.occurrences += 1
        self.evidence_msg_ids.update(fact.evidence_msg_ids)
        self.last_seen_at = max(self.last_seen_at, seen_at) if self.last_seen_at else seen_at
        if user_name:
            self.user_name = user_name

    @property
    def average_confidence(self) -> float:
        if self.occurrences <= 0:
            return 0.0
        return self.confidence_sum / self.occurrences

    @property
    def final_confidence(self) -> float:
        confidence = (self.average_confidence * 0.7) + (self.confidence_max * 0.3)
        confidence += min(0.12, 0.04 * max(0, self.occurrences - 1))
        return min(0.98, confidence)


def _build_profile_extractor_model() -> ChatOpenAI:
    return ChatOpenAI(
        model=plugin_config.openai_model,
        api_key=SecretStr(plugin_config.openai_token),
        base_url=plugin_config.openai_base_url,
        temperature=0.2,
    )


def _strip_id_lines(content: str) -> str:
    lines: list[str] = []
    for raw_line in (content or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("id:") or line.startswith("回复id:"):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _normalize_message_for_profile_context(message: ChatHistorySchema) -> str:
    text = _strip_id_lines(message.content or "")
    if message.content_type == "image":
        return "[发送图片]"
    if message.content_type == "bot" and "发送了图片" in text:
        return "[bot发送图片]"
    return text


def _message_token_cost(message: ChatHistorySchema) -> int:
    speaker = message.user_name or message.user_id
    content = _normalize_message_for_profile_context(message)
    if not content:
        return 0
    return estimate_token_count(f"{speaker}:{content}")


def _target_user_message_counts(
    messages: list[ChatHistorySchema],
    *,
    include_synced_targets: bool,
) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for message in messages:
        if message.content_type not in ("text", "image"):
            continue
        if include_synced_targets or not bool(message.mem0_synced):
            counts[str(message.user_id)] += 1
    return counts


def _window_target_users(
    window: list[ChatHistorySchema],
    *,
    include_synced_targets: bool,
    min_messages_per_user: int,
) -> dict[str, str]:
    target_user_counts = _target_user_message_counts(
        window,
        include_synced_targets=include_synced_targets,
    )
    users: dict[str, str] = {}
    for message in window:
        if message.content_type not in ("text", "image"):
            continue
        if not include_synced_targets and bool(message.mem0_synced):
            continue
        user_id = str(message.user_id)
        if target_user_counts.get(user_id, 0) < min_messages_per_user:
            continue
        users[user_id] = message.user_name or users.get(user_id, "")
    return users


def _normalize_fact_content(content: str) -> str:
    compact = re.sub(r"[，。；、,.!?！？\s]+", "", content or "")
    return compact.strip().lower()


def _fact_should_keep(fact: AggregatedProfileFact) -> bool:
    if fact.memory_type == "recent_topics":
        return fact.final_confidence >= MIN_RECENT_TOPIC_CONFIDENCE
    return fact.occurrences >= 2 or fact.final_confidence >= HIGH_CONFIDENCE_DIRECT_WRITE


def _extract_json_payload(text: str) -> str:
    body = (text or "").strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", body, re.S)
    if fence_match:
        return fence_match.group(1).strip()

    start = body.find("{")
    end = body.rfind("}")
    if start >= 0 and end > start:
        return body[start : end + 1]
    return body


def _serialize_datetime(value: datetime.datetime | None) -> str | None:
    if value is None:
        return None
    return value.isoformat()


def _build_window_transcript(window: list[ChatHistorySchema], target_user_id: str) -> tuple[str, list[int]]:
    lines: list[str] = []
    target_msg_ids: list[int] = []
    for message in window:
        content = _normalize_message_for_profile_context(message)
        if not content:
            continue

        role = "other"
        if message.content_type == "bot" or str(message.user_id) == str(plugin_config.bot_name):
            role = "bot"
        elif str(message.user_id) == target_user_id:
            role = "target"

        timestamp = message.created_at.strftime("%Y-%m-%d %H:%M:%S")
        speaker = message.user_name or message.user_id
        lines.append(f"[{message.msg_id}][{timestamp}][{role}] {speaker}: {content}")
        if role == "target":
            target_msg_ids.append(message.msg_id)

    return "\n".join(lines), target_msg_ids


async def list_pending_mem0_sessions(db_session: AsyncSession) -> list[str]:
    result = await db_session.execute(
        Select(ChatHistory.session_id)
        .where(
            ChatHistory.mem0_synced.is_(False),
            ChatHistory.content_type.in_(["text", "image"]),
        )
        .distinct()
    )
    return [str(session_id) for session_id in result.scalars().all() if session_id]


async def split_session_time_windows(
    db_session: AsyncSession,
    session_id: str,
    max_time_gap: timedelta = MAX_WINDOW_TIME_GAP,
    include_synced_targets: bool = False,
) -> list[list[ChatHistorySchema]]:
    query = (
        Select(ChatHistory)
        .where(
            ChatHistory.session_id == session_id,
            ChatHistory.content_type.in_(["text", "image", "bot"]),
        )
        .order_by(ChatHistory.created_at, ChatHistory.msg_id)
    )
    all_messages = (await db_session.execute(query)).scalars().all()
    all_messages = [ChatHistorySchema.model_validate(message) for message in all_messages]
    if not all_messages:
        return []

    windows: list[list[ChatHistorySchema]] = []
    current_window: list[ChatHistorySchema] = []
    last_message_time: datetime.datetime | None = None

    for message in all_messages:
        start_new_window = False
        if last_message_time and (message.created_at - last_message_time) > max_time_gap:
            start_new_window = True

        if start_new_window and current_window:
            windows.append(current_window)
            current_window = []

        current_window.append(message)
        last_message_time = message.created_at

    if current_window:
        windows.append(current_window)

    return [
        window
        for window in windows
        if _target_user_message_counts(
            window,
            include_synced_targets=include_synced_targets,
        )
    ]


def split_large_time_window(
    window: list[ChatHistorySchema],
    *,
    max_window_messages: int = MAX_WINDOW_MESSAGES,
    max_window_tokens: int = MAX_WINDOW_TOKENS,
) -> list[list[ChatHistorySchema]]:
    sub_windows: list[list[ChatHistorySchema]] = []
    current_window: list[ChatHistorySchema] = []
    current_tokens = 0

    for message in window:
        message_tokens = _message_token_cost(message)
        start_new_window = False
        if current_tokens + message_tokens > max_window_tokens:
            start_new_window = True
        elif len(current_window) >= max_window_messages:
            start_new_window = True

        if start_new_window and current_window:
            sub_windows.append(current_window)
            current_window = []
            current_tokens = 0

        current_window.append(message)
        current_tokens += message_tokens

    if current_window:
        sub_windows.append(current_window)
    return sub_windows


async def mark_messages_as_mem0_synced_batch(db_session: AsyncSession, msg_ids: list[int]) -> None:
    if not msg_ids:
        return
    await db_session.execute(
        Update(ChatHistory)
        .where(ChatHistory.msg_id.in_(msg_ids))
        .values(mem0_synced=True)
    )


async def _extract_profile_facts_from_window(
    model: ChatOpenAI,
    session_id: str,
    target_user_id: str,
    target_user_name: str,
    window: list[ChatHistorySchema],
) -> list[ProfileFactCandidate]:
    transcript, target_msg_ids = _build_window_transcript(window, target_user_id)
    if not transcript or not target_msg_ids:
        return []

    prompt = f"""你是QQ群用户画像抽取器。

任务：基于一段群聊上下文窗口，为目标用户提炼“适合写入长期画像”的候选事实。

目标用户：
- session_id: {session_id}
- user_id: {target_user_id}
- user_name: {target_user_name}
- bot_name: {plugin_config.bot_name}

规则：
1. 只分析目标用户，不要总结其他群友。
2. 只有下面 6 种 memory_type 可以输出：
   - speaking_style
   - interests
   - dislikes
   - relationship_tone
   - recent_topics
   - stable_traits
3. content 必须是简短中文短语，不要写完整句子，不要带主语，不要超过 20 个字。
4. 不要输出原始消息复述、图片描述、一次性事件、临时安排、链接、具体数字流水账。
5. speaking_style / interests / dislikes / relationship_tone / stable_traits 必须足够像“画像”，证据不足就不要写。
6. recent_topics 允许保留本窗口里目标用户最近在聊的话题，但不要把整个群的公共话题误写成该用户画像。
7. evidence_msg_ids 只能填写窗口里真正支持该结论的消息 ID，尽量精简。
8. 每类最多 2 条，总计最多 {MAX_FACTS_PER_WINDOW_USER} 条；证据不足时返回空数组。
9. 必须返回严格 JSON，不要解释，不要 Markdown。

输出格式：
{{
  "facts": [
    {{
      "memory_type": "speaking_style",
      "content": "短句吐槽",
      "confidence": 0.82,
      "evidence_msg_ids": [123, 128]
    }}
  ]
}}

群聊上下文窗口：
{transcript}
"""

    try:
        response = await model.ainvoke(prompt)
    except Exception as exc:
        logger.warning(
            f"mem0 profile extraction failed: session_id={session_id} user_id={target_user_id} "
            f"error={type(exc).__name__}: {exc}"
        )
        return []

    content = response.content if hasattr(response, "content") else response
    if not isinstance(content, str):
        content = str(content)

    try:
        payload = _extract_json_payload(content)
        parsed = ProfileExtractionResult.model_validate(json.loads(payload))
    except Exception as exc:
        logger.warning(
            f"mem0 profile parse failed: session_id={session_id} user_id={target_user_id} "
            f"error={type(exc).__name__}: {exc} raw={content}"
        )
        return []

    valid_msg_ids = {message.msg_id for message in window}
    facts: list[ProfileFactCandidate] = []
    per_type_counts = defaultdict(int)
    for fact in parsed.facts:
        fact_content = re.sub(r"\s+", "", fact.content or "").strip()
        if not fact_content:
            continue
        if per_type_counts[fact.memory_type] >= MAX_FACTS_PER_TYPE:
            continue
        fact_confidence = max(0.0, min(1.0, fact.confidence))
        if fact_confidence < MIN_STABLE_FACT_CONFIDENCE and fact.memory_type != "recent_topics":
            continue

        filtered_ids = [msg_id for msg_id in fact.evidence_msg_ids if msg_id in valid_msg_ids][:8]
        if not filtered_ids:
            filtered_ids = target_msg_ids[:4]

        facts.append(
            ProfileFactCandidate(
                memory_type=fact.memory_type,
                content=fact_content[:24],
                confidence=fact_confidence,
                evidence_msg_ids=filtered_ids,
            )
        )
        per_type_counts[fact.memory_type] += 1
        if len(facts) >= MAX_FACTS_PER_WINDOW_USER:
            break
    return facts


async def _write_aggregated_profile_facts(
    db_session: AsyncSession,
    session_id: str,
    user_id: str,
    user_name: str,
    facts: list[AggregatedProfileFact],
    msg_ids_to_mark: list[int],
) -> bool:
    for fact in facts:
        profile_text = f"{fact.memory_type}: {fact.content}"
        ok = await mem0_client.add_messages(
            user_id=user_id,
            user_name=user_name,
            session_id=session_id,
            messages=[{"role": "user", "content": f"用户长期画像事实：{profile_text}"}],
            source="ai_groupmate_profile_summary",
            extra_metadata={
                "memory_type": fact.memory_type,
                "confidence": round(fact.final_confidence, 4),
                "evidence_msg_ids": sorted(fact.evidence_msg_ids)[:20],
                "last_updated_at": _serialize_datetime(fact.last_seen_at),
            },
        )
        if not ok:
            logger.warning(
                f"mem0 profile write failed: session_id={session_id} user_id={user_id} "
                f"memory_type={fact.memory_type} content={fact.content}"
            )
            return False

    await mark_messages_as_mem0_synced_batch(db_session, msg_ids_to_mark)
    await db_session.commit()
    return True


async def process_and_sync_mem0_session_profile(
    db_session: AsyncSession,
    session_id: str,
) -> dict | None:
    if not mem0_client.enabled:
        return None

    time_windows = await split_session_time_windows(
        db_session,
        session_id,
        include_synced_targets=False,
    )
    if not time_windows:
        return None

    model = _build_profile_extractor_model()
    aggregated_by_user: dict[str, dict[tuple[str, str], AggregatedProfileFact]] = defaultdict(dict)
    user_names: dict[str, str] = {}
    user_msg_ids_to_mark: dict[str, set[int]] = defaultdict(set)
    processed_windows = 0
    skipped_low_activity_users = 0

    for time_window in time_windows:
        target_user_counts = _target_user_message_counts(
            time_window,
            include_synced_targets=False,
        )
        if not target_user_counts:
            continue

        target_users = _window_target_users(
            time_window,
            include_synced_targets=False,
            min_messages_per_user=MIN_USER_MESSAGES_PER_WINDOW,
        )
        if not target_users:
            for user_id, message_count in target_user_counts.items():
                if message_count >= MIN_USER_MESSAGES_PER_WINDOW:
                    continue
                skipped_low_activity_users += 1
                for message in time_window:
                    if (
                        message.content_type in ("text", "image")
                        and not bool(message.mem0_synced)
                        and str(message.user_id) == user_id
                    ):
                        user_msg_ids_to_mark[user_id].add(message.msg_id)
            continue

        for user_id, message_count in target_user_counts.items():
            if message_count >= MIN_USER_MESSAGES_PER_WINDOW:
                continue
            skipped_low_activity_users += 1
            for message in time_window:
                if (
                    message.content_type in ("text", "image")
                    and not bool(message.mem0_synced)
                    and str(message.user_id) == user_id
                ):
                    user_msg_ids_to_mark[user_id].add(message.msg_id)

        sub_windows = split_large_time_window(time_window)
        for window in sub_windows:
            processed_windows += 1
            for user_id, user_name in target_users.items():
                if not any(
                    message.content_type in ("text", "image") and str(message.user_id) == user_id
                    for message in window
                ):
                    continue

                user_names[user_id] = user_name or user_names.get(user_id, "")
                facts = await _extract_profile_facts_from_window(
                    model=model,
                    session_id=session_id,
                    target_user_id=user_id,
                    target_user_name=user_names[user_id] or user_name or user_id,
                    window=window,
                )
                seen_at = window[-1].created_at
                for fact in facts:
                    fact_key = (fact.memory_type, _normalize_fact_content(fact.content))
                    if not fact_key[1]:
                        continue
                    aggregate = aggregated_by_user[user_id].get(fact_key)
                    if not aggregate:
                        aggregate = AggregatedProfileFact(
                            memory_type=fact.memory_type,
                            content=fact.content,
                            user_name=user_names[user_id] or user_id,
                        )
                        aggregated_by_user[user_id][fact_key] = aggregate
                    aggregate.add(fact, user_names[user_id] or user_id, seen_at)

            for message in window:
                if (
                    message.content_type in ("text", "image")
                    and not bool(message.mem0_synced)
                    and str(message.user_id) in target_users
                ):
                    user_msg_ids_to_mark[str(message.user_id)].add(message.msg_id)

    synced_users = 0
    synced_messages = 0
    written_facts = 0
    for user_id in sorted(user_msg_ids_to_mark):
        fact_map = aggregated_by_user.get(user_id, {})
        final_facts = [fact for fact in fact_map.values() if _fact_should_keep(fact)]
        final_facts.sort(
            key=lambda item: (
                item.memory_type != "recent_topics",
                item.final_confidence,
                item.occurrences,
                item.last_seen_at or datetime.datetime.min,
            ),
            reverse=True,
        )
        final_facts = final_facts[:MAX_FINAL_FACTS_PER_USER]
        msg_ids_to_mark = sorted(user_msg_ids_to_mark.get(user_id, set()))
        if not msg_ids_to_mark:
            continue

        if not final_facts:
            await mark_messages_as_mem0_synced_batch(db_session, msg_ids_to_mark)
            await db_session.commit()
            synced_users += 1
            synced_messages += len(msg_ids_to_mark)
            continue

        ok = await _write_aggregated_profile_facts(
            db_session=db_session,
            session_id=session_id,
            user_id=user_id,
            user_name=user_names.get(user_id, user_id),
            facts=final_facts,
            msg_ids_to_mark=msg_ids_to_mark,
        )
        if not ok:
            await db_session.rollback()
            continue

        synced_users += 1
        synced_messages += len(msg_ids_to_mark)
        written_facts += len(final_facts)

    return {
        "session_id": session_id,
        "processed_time_windows": len(time_windows),
        "processed_windows": processed_windows,
        "skipped_low_activity_users": skipped_low_activity_users,
        "synced_users": synced_users,
        "synced_messages": synced_messages,
        "written_facts": written_facts,
    }
