#!/usr/bin/env python3
"""Backfill user profile memories into mem0 from PostgreSQL chat history."""

from __future__ import annotations

import argparse
import datetime as dt
import difflib
import json
import os
import re
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal

try:
    import psycopg
    from psycopg.rows import dict_row
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency: psycopg\n"
        "Install with: python3 -m pip install 'psycopg[binary]'"
    ) from exc

try:
    from langchain_openai import ChatOpenAI
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency: langchain_openai\n"
        "Install with: python3 -m pip install langchain-openai"
    ) from exc

try:
    from pydantic import BaseModel, Field, SecretStr
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency: pydantic\n"
        "Install with: python3 -m pip install pydantic"
    ) from exc


DEFAULT_TABLE = "nonebot_plugin_ai_groupmate_chathistory"
TABLE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?$")
SUPPORTED_CONTENT_TYPES = ("text", "image")
SUPPORTED_CONTEXT_TYPES = ("text", "image", "bot")
DEFAULT_SOURCE = "ai_groupmate_profile_summary_backfill"
DEFAULT_MEM0_TIMEOUT_SECONDS = 120.0
DEFAULT_WINDOW_GAP_MINUTES = 10
DEFAULT_WINDOW_MESSAGES = 120
DEFAULT_WINDOW_TOKENS = 5000
DEFAULT_MIN_USER_MESSAGES_PER_WINDOW = 30
DEFAULT_RECENT_TOPICS_MAX_AGE_DAYS = 3
MAX_FACTS_PER_WINDOW_USER = 6
MAX_FACTS_PER_TYPE = 2
MAX_FINAL_FACTS_PER_USER = 8
MIN_RECENT_TOPIC_CONFIDENCE = 0.60
MIN_STABLE_FACT_CONFIDENCE = 0.72
HIGH_CONFIDENCE_DIRECT_WRITE = 0.86


def estimate_token_count(text: str) -> int:
    try:
        import tiktoken

        encoder = tiktoken.get_encoding("cl100k_base")
        return len(encoder.encode(text))
    except Exception:
        compact = re.sub(r"\s+", "", text)
        return max(len(compact), len(text) // 2)


@dataclass
class ChatRow:
    msg_id: int
    session_id: str
    user_id: str
    content_type: str
    content: str
    created_at: dt.datetime
    user_name: str | None
    mem0_synced: bool


class ProfileFactCandidate(BaseModel):
    memory_type: Literal[
        "speaking_style",
        "interests",
        "dislikes",
        "relationship_tone",
        "recent_topics",
        "stable_traits",
    ] = Field(description="画像类型")
    content: str = Field(description="画像内容，必须是简短中文短语。")
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
    last_seen_at: dt.datetime | None = None

    def add(self, fact: ProfileFactCandidate, user_name: str, seen_at: dt.datetime) -> None:
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


class Mem0ApiClient:
    def __init__(self, base_url: str, api_token: str = "", timeout: float = DEFAULT_MEM0_TIMEOUT_SECONDS):
        self.base_url = base_url.rstrip("/")
        self.api_token = api_token.strip()
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
        return headers

    def _request_json(self, method: str, path: str, payload: dict[str, Any]) -> Any:
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=self._headers(),
            method=method.upper(),
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8")
            except Exception:
                pass
            raise RuntimeError(f"mem0 http error {exc.code}: {detail or exc.reason}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"mem0 request failed: {exc}") from exc

        if not body:
            return None
        return json.loads(body)

    def add_profile_fact(
        self,
        *,
        user_id: str,
        user_name: str,
        session_id: str,
        memory_type: str,
        content: str,
        confidence: float,
        evidence_msg_ids: list[int],
        last_updated_at: str | None,
        source: str,
    ) -> Any:
        payload = {
            "user_id": user_id,
            "messages": [{"role": "user", "content": _profile_fact_message(memory_type, content)}],
            "metadata": {
                "session_id": session_id,
                "user_name": user_name,
                "source": source,
                "memory_type": memory_type,
                "confidence": round(confidence, 4),
                "evidence_msg_ids": evidence_msg_ids,
                "last_updated_at": last_updated_at,
            },
        }
        return self._request_json("POST", "/memories", payload)


def _normalize_db_url(url: str) -> str:
    value = (url or "").strip()
    value = value.replace("postgresql+asyncpg://", "postgresql://", 1)
    if value.startswith("postgres://"):
        value = "postgresql://" + value[len("postgres://") :]
    return value


def _parse_env_file(env_file: str | None) -> dict[str, str]:
    if not env_file:
        return {}
    path = Path(env_file)
    if not path.exists():
        raise SystemExit(f"env file not found: {env_file}")

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if not key:
            continue

        if raw_value[:1] in {"'", '"'}:
            quote = raw_value[0]
            end = raw_value.rfind(quote)
            if end > 0:
                value = raw_value[1:end]
            else:
                value = raw_value[1:]
        else:
            value = raw_value.split("#", 1)[0].strip()
        values[key] = value
    return values


def _pick_value(*values: str | None) -> str:
    for value in values:
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


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


def _normalize_message_for_profile_context(message: ChatRow) -> str:
    text = _strip_id_lines(message.content)
    if message.content_type == "image":
        return "[发送图片]"
    if message.content_type == "bot" and "发送了图片" in text:
        return "[bot发送图片]"
    return text


def _message_token_cost(message: ChatRow) -> int:
    content = _normalize_message_for_profile_context(message)
    if not content:
        return 0
    speaker = message.user_name or message.user_id
    return estimate_token_count(f"{speaker}:{content}")


def _target_user_message_counts(
    messages: list[ChatRow],
    *,
    include_synced_targets: bool,
    target_user_id: str,
) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for message in messages:
        if message.content_type not in SUPPORTED_CONTENT_TYPES:
            continue
        user_id = str(message.user_id)
        if target_user_id and user_id != target_user_id:
            continue
        if include_synced_targets or not message.mem0_synced:
            counts[user_id] += 1
    return counts


def _window_target_users(
    window: list[ChatRow],
    *,
    include_synced_targets: bool,
    target_user_id: str,
    min_messages_per_user: int,
) -> dict[str, str]:
    target_user_counts = _target_user_message_counts(
        window,
        include_synced_targets=include_synced_targets,
        target_user_id=target_user_id,
    )
    users: dict[str, str] = {}
    for message in window:
        if message.content_type not in SUPPORTED_CONTENT_TYPES:
            continue
        user_id = str(message.user_id)
        if target_user_id and user_id != target_user_id:
            continue
        if not include_synced_targets and message.mem0_synced:
            continue
        if target_user_counts.get(user_id, 0) < min_messages_per_user:
            continue
        users[user_id] = message.user_name or users.get(user_id, "")
    return users


def _normalize_fact_content(content: str) -> str:
    compact = re.sub(r"[，。；、,.!?！？\s]+", "", content or "")
    return compact.strip().lower()


def _profile_fact_message(memory_type: str, content: str) -> str:
    if memory_type == "interests":
        return f"我喜欢{content}"
    if memory_type == "dislikes":
        return f"我不喜欢{content}"
    if memory_type == "speaking_style":
        return f"我的说话风格偏{content}"
    if memory_type == "relationship_tone":
        return f"我和bot互动时通常是{content}"
    if memory_type == "recent_topics":
        return f"我最近在聊{content}"
    return f"我的稳定特征是{content}"


def _facts_are_similar(left: str, right: str) -> bool:
    left_norm = _normalize_fact_content(left)
    right_norm = _normalize_fact_content(right)
    if not left_norm or not right_norm:
        return False
    if left_norm == right_norm:
        return True
    if left_norm in right_norm or right_norm in left_norm:
        return True
    return difflib.SequenceMatcher(None, left_norm, right_norm).ratio() >= 0.72


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


def _serialize_datetime(value: dt.datetime | None) -> str | None:
    if value is None:
        return None
    return value.isoformat()


def _window_end_time(window: list[ChatRow]) -> dt.datetime | None:
    if not window:
        return None
    return window[-1].created_at


def _window_allows_recent_topics(
    window: list[ChatRow],
    *,
    recent_topics_max_age_days: int,
) -> bool:
    if recent_topics_max_age_days <= 0:
        return True
    window_end = _window_end_time(window)
    if window_end is None:
        return False

    if window_end.tzinfo is None:
        now = dt.datetime.now()
    else:
        now = dt.datetime.now(dt.timezone.utc)
        window_end = window_end.astimezone(dt.timezone.utc)
    cutoff = now - timedelta(days=recent_topics_max_age_days)
    return window_end >= cutoff


def _build_window_transcript(window: list[ChatRow], target_user_id: str, bot_name: str) -> tuple[str, list[int]]:
    lines: list[str] = []
    target_msg_ids: list[int] = []
    for message in window:
        content = _normalize_message_for_profile_context(message)
        if not content:
            continue

        role = "other"
        if message.content_type == "bot" or str(message.user_id) == bot_name:
            role = "bot"
        elif str(message.user_id) == target_user_id:
            role = "target"

        timestamp = message.created_at.strftime("%Y-%m-%d %H:%M:%S")
        speaker = message.user_name or message.user_id
        lines.append(f"[{message.msg_id}][{timestamp}][{role}] {speaker}: {content}")
        if role == "target":
            target_msg_ids.append(message.msg_id)

    return "\n".join(lines), target_msg_ids


def ensure_mem0_synced_column(conn: psycopg.Connection[Any], table_name: str) -> None:
    index_name = f"ix_{table_name.replace('.', '_')}_mem0_synced"
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s)", (table_name,))
        exists = cur.fetchone()
        if not exists or not exists[0]:
            raise RuntimeError(f"table not found: {table_name}")

        cur.execute(
            f"""
            ALTER TABLE {table_name}
            ADD COLUMN IF NOT EXISTS mem0_synced BOOLEAN NOT NULL DEFAULT FALSE
            """
        )
        cur.execute(
            f"""
            CREATE INDEX IF NOT EXISTS {index_name}
            ON {table_name} (mem0_synced)
            """
        )
    conn.commit()


def list_target_sessions(conn: psycopg.Connection[Any], args: argparse.Namespace) -> list[str]:
    filters = [
        "session_id IS NOT NULL",
        "session_id <> ''",
        "content_type = ANY(%s)",
    ]
    params: list[Any] = [list(SUPPORTED_CONTENT_TYPES)]

    if args.session_id:
        filters.append("session_id = %s")
        params.append(args.session_id)
    if args.user_id:
        filters.append("user_id = %s")
        params.append(args.user_id)
    if not args.force:
        filters.append("COALESCE(mem0_synced, FALSE) = FALSE")

    query = f"""
        SELECT DISTINCT session_id
        FROM {args.table}
        WHERE {' AND '.join(filters)}
        ORDER BY session_id
    """
    if args.limit_sessions:
        query += " LIMIT %s"
        params.append(args.limit_sessions)

    with conn.cursor() as cur:
        cur.execute(query, params)
        return [str(row[0]) for row in cur.fetchall() if row[0]]


def load_session_rows(
    conn: psycopg.Connection[Any],
    *,
    table_name: str,
    session_id: str,
) -> list[ChatRow]:
    query = f"""
        SELECT msg_id, session_id, user_id, content_type, content, created_at, user_name, mem0_synced
        FROM {table_name}
        WHERE session_id = %s
          AND content_type = ANY(%s)
        ORDER BY created_at, msg_id
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query, (session_id, list(SUPPORTED_CONTEXT_TYPES)))
        rows = cur.fetchall()

    return [
        ChatRow(
            msg_id=int(row["msg_id"]),
            session_id=str(row["session_id"]),
            user_id=str(row["user_id"]),
            content_type=str(row["content_type"]),
            content=str(row["content"] or ""),
            created_at=row["created_at"],
            user_name=str(row["user_name"]) if row["user_name"] is not None else None,
            mem0_synced=bool(row["mem0_synced"]),
        )
        for row in rows
    ]


def split_session_time_windows(
    rows: list[ChatRow],
    *,
    max_time_gap: timedelta,
    include_synced_targets: bool,
    target_user_id: str,
) -> list[list[ChatRow]]:
    windows: list[list[ChatRow]] = []
    current_window: list[ChatRow] = []
    last_message_time: dt.datetime | None = None

    for message in rows:
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
            target_user_id=target_user_id,
        )
    ]


def split_large_time_window(
    window: list[ChatRow],
    *,
    max_window_messages: int,
    max_window_tokens: int,
) -> list[list[ChatRow]]:
    sub_windows: list[list[ChatRow]] = []
    current_window: list[ChatRow] = []
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


def mark_rows_synced(conn: psycopg.Connection[Any], table_name: str, msg_ids: list[int]) -> None:
    if not msg_ids:
        return
    with conn.cursor() as cur:
        cur.execute(
            f"UPDATE {table_name} SET mem0_synced = TRUE WHERE msg_id = ANY(%s)",
            (msg_ids,),
        )
    conn.commit()


def build_profile_extractor_model(args: argparse.Namespace) -> ChatOpenAI:
    return ChatOpenAI(
        model=args.openai_model,
        api_key=SecretStr(args.openai_token),
        base_url=args.openai_base_url,
        temperature=0.2,
    )


def extract_profile_facts_from_window(
    *,
    model: ChatOpenAI,
    session_id: str,
    target_user_id: str,
    target_user_name: str,
    bot_name: str,
    window: list[ChatRow],
    allow_recent_topics: bool,
) -> list[ProfileFactCandidate]:
    transcript, target_msg_ids = _build_window_transcript(window, target_user_id, bot_name)
    if not transcript or not target_msg_ids:
        return []

    window_ended_at = _serialize_datetime(_window_end_time(window)) or "unknown"
    recent_topics_rule = (
        "6. recent_topics 允许保留本窗口里目标用户最近在聊的话题，但不要把整个群的公共话题误写成该用户画像。"
        if allow_recent_topics
        else "6. recent_topics 只适用于距离当前现实时间 3 天内的窗口。当前窗口已经过期，禁止输出 recent_topics；请优先总结 speaking_style、interests、dislikes、relationship_tone、stable_traits。"
    )

    prompt = f"""你是QQ群用户画像抽取器。

任务：基于一段群聊上下文窗口，为目标用户提炼“适合写入长期画像”的候选事实。

目标用户：
- session_id: {session_id}
- user_id: {target_user_id}
- user_name: {target_user_name}
- bot_name: {bot_name}
- window_ended_at: {window_ended_at}

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
{recent_topics_rule}
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

    response = model.invoke(prompt)
    content = response.content if hasattr(response, "content") else response
    if not isinstance(content, str):
        content = str(content)

    payload = _extract_json_payload(content)
    parsed = ProfileExtractionResult.model_validate(json.loads(payload))

    valid_msg_ids = {message.msg_id for message in window}
    facts: list[ProfileFactCandidate] = []
    per_type_counts = defaultdict(int)
    for fact in parsed.facts:
        fact_content = re.sub(r"\s+", "", fact.content or "").strip()
        if not fact_content:
            continue
        if fact.memory_type == "recent_topics" and not allow_recent_topics:
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


def converge_profile_facts(
    *,
    model: ChatOpenAI,
    session_id: str,
    user_id: str,
    user_name: str,
    facts: list[AggregatedProfileFact],
) -> list[AggregatedProfileFact]:
    if len(facts) <= 1:
        return facts[:MAX_FINAL_FACTS_PER_USER]

    candidate_lines = []
    for index, fact in enumerate(facts[:20], start=1):
        candidate_lines.append(
            json.dumps(
                {
                    "index": index,
                    "memory_type": fact.memory_type,
                    "content": fact.content,
                    "confidence": round(fact.final_confidence, 4),
                    "occurrences": fact.occurrences,
                    "evidence_msg_ids": sorted(fact.evidence_msg_ids)[:12],
                    "last_updated_at": _serialize_datetime(fact.last_seen_at),
                },
                ensure_ascii=False,
            )
        )

    prompt = f"""你是用户画像事实收敛器。

任务：把同一个用户的一批候选画像事实做去重、合并、收敛，输出最终画像。

目标用户：
- session_id: {session_id}
- user_id: {user_id}
- user_name: {user_name}

规则：
1. 合并近义重复、同义词和强相关变体。
2. interests / recent_topics 里像“玩音游osu!”、“音游（osu!mania）”这种高度相关表述，尽量收敛成 1 条更自然的说法。
3. 不要输出事件片段、图片描述、流水账。
4. 每种 memory_type 最多 2 条，总计最多 {MAX_FINAL_FACTS_PER_USER} 条。
5. content 必须是简短中文短语，不超过 20 个字。
6. evidence_msg_ids 只保留最能支撑该结论的几个。
7. 必须返回严格 JSON，不要解释。

候选事实：
{chr(10).join(candidate_lines)}

返回格式：
{{
  "facts": [
    {{
      "memory_type": "interests",
      "content": "玩音游osu",
      "confidence": 0.9,
      "evidence_msg_ids": [123, 125]
    }}
  ]
}}
"""

    try:
        response = model.invoke(prompt)
        content = response.content if hasattr(response, "content") else response
        if not isinstance(content, str):
            content = str(content)
        parsed = ProfileExtractionResult.model_validate(json.loads(_extract_json_payload(content)))
    except Exception as exc:
        print(
            f"  converge_failed session_id={session_id} user_id={user_id} "
            f"error={type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return facts[:MAX_FINAL_FACTS_PER_USER]

    converged: list[AggregatedProfileFact] = []
    per_type_counts = defaultdict(int)
    for fact in parsed.facts:
        if per_type_counts[fact.memory_type] >= MAX_FACTS_PER_TYPE:
            continue

        matched_sources = [
            source
            for source in facts
            if source.memory_type == fact.memory_type and _facts_are_similar(source.content, fact.content)
        ]
        evidence_msg_ids: set[int] = set(fact.evidence_msg_ids)
        latest_seen_at: dt.datetime | None = None
        source_best_confidence = fact.confidence
        for source in matched_sources:
            evidence_msg_ids.update(source.evidence_msg_ids)
            if source.last_seen_at and (latest_seen_at is None or source.last_seen_at > latest_seen_at):
                latest_seen_at = source.last_seen_at
            source_best_confidence = max(source_best_confidence, source.final_confidence)

        converged.append(
            AggregatedProfileFact(
                memory_type=fact.memory_type,
                content=fact.content[:24],
                user_name=user_name,
                confidence_sum=source_best_confidence,
                confidence_max=source_best_confidence,
                occurrences=max(1, len(matched_sources)),
                evidence_msg_ids=evidence_msg_ids,
                last_seen_at=latest_seen_at,
            )
        )
        per_type_counts[fact.memory_type] += 1
        if len(converged) >= MAX_FINAL_FACTS_PER_USER:
            break

    return converged or facts[:MAX_FINAL_FACTS_PER_USER]


def process_session(
    conn: psycopg.Connection[Any],
    *,
    args: argparse.Namespace,
    model: ChatOpenAI,
    mem0_client: Mem0ApiClient | None,
    session_id: str,
) -> dict[str, Any]:
    rows = load_session_rows(conn, table_name=args.table, session_id=session_id)
    time_windows = split_session_time_windows(
        rows,
        max_time_gap=timedelta(minutes=args.max_time_gap_minutes),
        include_synced_targets=args.force,
        target_user_id=args.user_id,
    )
    total_available_time_windows = len(time_windows)
    if args.limit_time_windows > 0:
        if args.time_window_order == "desc":
            time_windows = time_windows[-args.limit_time_windows :]
        else:
            time_windows = time_windows[: args.limit_time_windows]
    if args.time_window_order == "desc":
        time_windows = list(reversed(time_windows))
    if not time_windows:
        return {
            "session_id": session_id,
            "available_time_windows": total_available_time_windows,
            "processed_time_windows": 0,
            "processed_windows": 0,
            "skipped_low_activity_users": 0,
            "synced_users": 0,
            "synced_messages": 0,
            "written_facts": 0,
        }

    aggregated_by_user: dict[str, dict[tuple[str, str], AggregatedProfileFact]] = defaultdict(dict)
    user_names: dict[str, str] = {}
    user_msg_ids_to_mark: dict[str, set[int]] = defaultdict(set)
    processed_windows = 0
    skipped_low_activity_users = 0

    for time_window_index, time_window in enumerate(time_windows, start=1):
        target_user_counts = _target_user_message_counts(
            time_window,
            include_synced_targets=args.force,
            target_user_id=args.user_id,
        )
        if not target_user_counts:
            continue

        target_users = _window_target_users(
            time_window,
            include_synced_targets=args.force,
            target_user_id=args.user_id,
            min_messages_per_user=args.min_user_messages_per_window,
        )

        for user_id, message_count in target_user_counts.items():
            if message_count >= args.min_user_messages_per_window:
                continue
            skipped_low_activity_users += 1
            for message in time_window:
                if message.content_type not in SUPPORTED_CONTENT_TYPES:
                    continue
                if str(message.user_id) != user_id:
                    continue
                if not args.force and message.mem0_synced:
                    continue
                user_msg_ids_to_mark[user_id].add(message.msg_id)

        if not target_users:
            continue

        sub_windows = split_large_time_window(
            time_window,
            max_window_messages=args.max_window_messages,
            max_window_tokens=args.max_window_tokens,
        )
        allow_recent_topics = _window_allows_recent_topics(
            time_window,
            recent_topics_max_age_days=args.recent_topics_max_age_days,
        )
        print(
            f"  time_window={time_window_index}/{len(time_windows)} "
            f"qualified_users={len(target_users)} sub_windows={len(sub_windows)} "
            f"allow_recent_topics={str(allow_recent_topics).lower()}"
        )

        for window in sub_windows:
            processed_windows += 1
            for user_id, user_name in target_users.items():
                if not any(
                    message.content_type in SUPPORTED_CONTENT_TYPES and str(message.user_id) == user_id
                    for message in window
                ):
                    continue

                user_names[user_id] = user_name or user_names.get(user_id, "")
                try:
                    facts = extract_profile_facts_from_window(
                        model=model,
                        session_id=session_id,
                        target_user_id=user_id,
                        target_user_name=user_names[user_id] or user_id,
                        bot_name=args.bot_name,
                        window=window,
                        allow_recent_topics=allow_recent_topics,
                    )
                except Exception as exc:
                    print(
                        f"  extract_failed session_id={session_id} user_id={user_id} "
                        f"error={type(exc).__name__}: {exc}",
                        file=sys.stderr,
                    )
                    continue

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
                if message.content_type not in SUPPORTED_CONTENT_TYPES:
                    continue
                if not args.force and message.mem0_synced:
                    continue
                if str(message.user_id) not in target_users:
                    continue
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
                item.last_seen_at or dt.datetime.min,
            ),
            reverse=True,
        )
        final_facts = converge_profile_facts(
            model=model,
            session_id=session_id,
            user_id=user_id,
            user_name=user_names.get(user_id, user_id),
            facts=final_facts[:20],
        )[:MAX_FINAL_FACTS_PER_USER]
        msg_ids_to_mark = sorted(user_msg_ids_to_mark.get(user_id, set()))
        if not msg_ids_to_mark:
            continue

        if final_facts:
            print(
                f"  user_id={user_id} pending_msgs={len(msg_ids_to_mark)} "
                f"facts={len(final_facts)}"
            )

        for fact in final_facts:
            print(
                f"    fact {fact.memory_type} | {fact.content} "
                f"| conf={fact.final_confidence:.3f} | occurs={fact.occurrences} "
                f"| evidence={sorted(fact.evidence_msg_ids)[:8]}"
            )

        if args.dry_run:
            continue

        if final_facts:
            if mem0_client is None:
                raise RuntimeError("mem0_client is not initialized")
            for fact in final_facts:
                mem0_client.add_profile_fact(
                    user_id=user_id,
                    user_name=user_names.get(user_id, user_id),
                    session_id=session_id,
                    memory_type=fact.memory_type,
                    content=fact.content,
                    confidence=fact.final_confidence,
                    evidence_msg_ids=sorted(fact.evidence_msg_ids)[:20],
                    last_updated_at=_serialize_datetime(fact.last_seen_at),
                    source=args.source,
                )
            written_facts += len(final_facts)

        if not args.no_mark_synced:
            mark_rows_synced(conn, args.table, msg_ids_to_mark)
            synced_users += 1
            synced_messages += len(msg_ids_to_mark)

    return {
        "session_id": session_id,
        "available_time_windows": total_available_time_windows,
        "processed_time_windows": len(time_windows),
        "processed_windows": processed_windows,
        "skipped_low_activity_users": skipped_low_activity_users,
        "synced_users": synced_users,
        "synced_messages": synced_messages,
        "written_facts": written_facts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill user profile memories into mem0 from PostgreSQL chat history."
    )
    parser.add_argument("--env-file", default="")
    parser.add_argument("--database-url", default="")
    parser.add_argument("--mem0-base-url", default="")
    parser.add_argument("--mem0-api-token", default="")
    parser.add_argument("--mem0-timeout-seconds", type=float, default=DEFAULT_MEM0_TIMEOUT_SECONDS)
    parser.add_argument("--openai-base-url", default="")
    parser.add_argument("--openai-model", default="")
    parser.add_argument("--openai-token", default="")
    parser.add_argument("--bot-name", default="")
    parser.add_argument("--table", default=DEFAULT_TABLE)
    parser.add_argument("--session-id", default="")
    parser.add_argument("--user-id", default="")
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--max-time-gap-minutes", type=int, default=DEFAULT_WINDOW_GAP_MINUTES)
    parser.add_argument("--max-window-messages", type=int, default=DEFAULT_WINDOW_MESSAGES)
    parser.add_argument("--max-window-tokens", type=int, default=DEFAULT_WINDOW_TOKENS)
    parser.add_argument("--min-user-messages-per-window", type=int, default=DEFAULT_MIN_USER_MESSAGES_PER_WINDOW)
    parser.add_argument("--recent-topics-max-age-days", type=int, default=DEFAULT_RECENT_TOPICS_MAX_AGE_DAYS)
    parser.add_argument("--limit-time-windows", type=int, default=0)
    parser.add_argument("--time-window-order", choices=("asc", "desc"), default="asc")
    parser.add_argument("--limit-sessions", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="Include rows already marked mem0_synced=true.")
    parser.add_argument("--dry-run", action="store_true", help="Only print extracted profile facts.")
    parser.add_argument("--no-mark-synced", action="store_true", help="Do not update mem0_synced in PostgreSQL.")
    args = parser.parse_args()

    if not TABLE_NAME_RE.match(args.table):
        parser.error("--table only supports letters, digits, underscores, and one optional schema prefix")
    if args.max_time_gap_minutes <= 0:
        parser.error("--max-time-gap-minutes must be > 0")
    if args.max_window_messages <= 0:
        parser.error("--max-window-messages must be > 0")
    if args.max_window_tokens <= 0:
        parser.error("--max-window-tokens must be > 0")
    if args.min_user_messages_per_window <= 0:
        parser.error("--min-user-messages-per-window must be > 0")
    if args.recent_topics_max_age_days < 0:
        parser.error("--recent-topics-max-age-days must be >= 0")
    if args.limit_time_windows < 0:
        parser.error("--limit-time-windows must be >= 0")
    if args.limit_sessions < 0:
        parser.error("--limit-sessions must be >= 0")
    if args.mem0_timeout_seconds <= 0:
        parser.error("--mem0-timeout-seconds must be > 0")

    env_values = _parse_env_file(args.env_file)
    args.database_url = _normalize_db_url(
        _pick_value(
            args.database_url,
            env_values.get("SQLALCHEMY_DATABASE_URL"),
            env_values.get("tortoise_orm_db_url"),
            os.getenv("DATABASE_URL"),
            os.getenv("SQLALCHEMY_DATABASE_URL"),
        )
    )
    args.mem0_base_url = _pick_value(
        args.mem0_base_url,
        env_values.get("ai_groupmate__mem0_api_base_url"),
        os.getenv("MEM0_API_BASE_URL"),
    )
    args.mem0_api_token = _pick_value(
        args.mem0_api_token,
        env_values.get("ai_groupmate__mem0_api_token"),
        os.getenv("MEM0_API_TOKEN"),
    )
    args.openai_base_url = _pick_value(
        args.openai_base_url,
        env_values.get("ai_groupmate__openai_base_url"),
        os.getenv("OPENAI_BASE_URL"),
    )
    args.openai_model = _pick_value(
        args.openai_model,
        env_values.get("ai_groupmate__openai_model"),
        os.getenv("OPENAI_MODEL"),
    )
    args.openai_token = _pick_value(
        args.openai_token,
        env_values.get("ai_groupmate__openai_token"),
        os.getenv("OPENAI_TOKEN"),
    )
    args.bot_name = _pick_value(
        args.bot_name,
        env_values.get("ai_groupmate__bot_name"),
        os.getenv("AI_GROUPMATE_BOT_NAME"),
    )

    if not args.database_url:
        parser.error("missing database url: use --database-url or --env-file")
    if not args.openai_base_url or not args.openai_model or not args.openai_token:
        parser.error("missing OpenAI config: use --openai-base-url/--openai-model/--openai-token or --env-file")
    if not args.bot_name:
        parser.error("missing bot name: use --bot-name or --env-file")
    if not args.dry_run and not args.mem0_base_url:
        parser.error("missing mem0 base url: use --mem0-base-url or --env-file")

    return args


def main() -> int:
    args = parse_args()
    model = build_profile_extractor_model(args)
    mem0_client = None if args.dry_run else Mem0ApiClient(
        base_url=args.mem0_base_url,
        api_token=args.mem0_api_token,
        timeout=args.mem0_timeout_seconds,
    )

    with psycopg.connect(args.database_url) as conn:
        ensure_mem0_synced_column(conn, args.table)
        sessions = list_target_sessions(conn, args)
        print(f"Target sessions: {len(sessions)}")

        total_windows = 0
        total_time_windows = 0
        total_available_time_windows = 0
        total_skipped_low_activity_users = 0
        total_users = 0
        total_messages = 0
        total_facts = 0

        for index, session_id in enumerate(sessions, start=1):
            print(f"[{index}/{len(sessions)}] session_id={session_id}")
            try:
                result = process_session(
                    conn,
                    args=args,
                    model=model,
                    mem0_client=mem0_client,
                    session_id=session_id,
                )
            except Exception as exc:
                print(
                    f"  session_failed session_id={session_id} error={type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                continue

            print(
                f"  time_windows={result['processed_time_windows']}/{result['available_time_windows']} "
                f"windows={result['processed_windows']} "
                f"skipped_low_activity_users={result['skipped_low_activity_users']} "
                f"synced_users={result['synced_users']} "
                f"synced_messages={result['synced_messages']} written_facts={result['written_facts']}"
            )
            total_available_time_windows += result["available_time_windows"]
            total_time_windows += result["processed_time_windows"]
            total_windows += result["processed_windows"]
            total_skipped_low_activity_users += result["skipped_low_activity_users"]
            total_users += result["synced_users"]
            total_messages += result["synced_messages"]
            total_facts += result["written_facts"]

        print(
            f"Done. sessions={len(sessions)} time_windows={total_time_windows}/{total_available_time_windows} "
            f"windows={total_windows} "
            f"skipped_low_activity_users={total_skipped_low_activity_users} "
            f"synced_users={total_users} synced_messages={total_messages} written_facts={total_facts}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
