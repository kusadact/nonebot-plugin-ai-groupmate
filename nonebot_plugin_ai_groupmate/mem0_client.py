import asyncio
import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from nonebot import get_plugin_config
from nonebot.log import logger

from .config import Config

plugin_config = get_plugin_config(Config).ai_groupmate


class Mem0Client:
    def __init__(
        self,
        base_url: str = "",
        api_token: str = "",
        timeout: float = 12.0,
        default_limit: int = 6,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_token = api_token.strip()
        self.timeout = timeout
        self.default_limit = default_limit

    @property
    def enabled(self) -> bool:
        return bool(self.base_url)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
        return headers

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        query: dict[str, Any] | None = None,
    ) -> Any:
        if not self.base_url:
            raise RuntimeError("mem0_api_base_url is empty")

        url = f"{self.base_url}{path}"
        if query:
            encoded_query = urllib.parse.urlencode(
                {k: v for k, v in query.items() if v is not None},
                doseq=True,
            )
            if encoded_query:
                url = f"{url}?{encoded_query}"

        data = None
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        request = urllib.request.Request(
            url,
            data=data,
            headers=self._headers(),
            method=method.upper(),
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8")
            except Exception:
                pass
            raise RuntimeError(f"mem0 http error {exc.code}: {body or exc.reason}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"mem0 request failed: {exc}") from exc

        if not body:
            return None
        return json.loads(body)

    async def search_memories(
        self,
        query: str,
        user_id: str,
        session_id: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        if not self.enabled or not user_id or not query.strip():
            return []

        payload: dict[str, Any] = {
            "query": query,
            "user_id": user_id,
            "limit": limit or self.default_limit,
        }
        if session_id:
            payload["filters"] = {"session_id": session_id}

        try:
            response = await asyncio.to_thread(self._request_json, "POST", "/search", payload, None)
        except Exception as exc:
            logger.warning(f"mem0 search failed: {type(exc).__name__}: {exc}")
            return []

        if isinstance(response, dict):
            results = response.get("results", [])
            if isinstance(results, list):
                return [item for item in results if isinstance(item, dict)]
        return []

    async def add_messages(
        self,
        user_id: str,
        user_name: str | None,
        session_id: str,
        messages: list[dict[str, str]],
        source: str = "ai_groupmate_chat_history",
        extra_metadata: dict[str, Any] | None = None,
    ) -> bool:
        if not self.enabled:
            return False
        normalized_messages = []
        for item in messages:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "").strip()
            content = str(item.get("content") or "").strip()
            if not role or not content:
                continue
            normalized_messages.append({"role": role, "content": content})

        if not user_id or not session_id or not normalized_messages:
            return False

        metadata = {
            "session_id": session_id,
            "user_name": user_name or "",
            "source": source,
        }
        if extra_metadata:
            metadata.update({k: v for k, v in extra_metadata.items() if v is not None})

        payload = {
            "user_id": user_id,
            "metadata": metadata,
            "messages": normalized_messages,
        }
        try:
            await asyncio.to_thread(self._request_json, "POST", "/memories", payload, None)
            return True
        except Exception as exc:
            logger.warning(f"mem0 add failed: {type(exc).__name__}: {exc}")
            return False


mem0_client = Mem0Client(
    base_url=plugin_config.mem0_api_base_url,
    api_token=plugin_config.mem0_api_token,
    timeout=plugin_config.mem0_timeout_seconds,
    default_limit=plugin_config.mem0_profile_limit,
)
