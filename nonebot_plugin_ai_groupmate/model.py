from datetime import datetime

from pydantic import BaseModel
from sqlalchemy import JSON, String, Boolean, Float, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column
from nonebot_plugin_orm import Model
from .favorability import status_desc_from_raw


class MediaStorage(Model):
    """媒体资源中心化存储"""

    media_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    file_hash: Mapped[str] = mapped_column(String(64), unique=True)  # SHA-256哈希
    file_path: Mapped[str]  # 实际存储路径或URL
    created_at: Mapped[datetime] = mapped_column(default=datetime.now, index=True)
    references: Mapped[int] = mapped_column(default=1, index=True)  # 引用计数
    description: Mapped[str]
    vectorized: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    blocked: Mapped[bool] = mapped_column(Boolean, default=False, index=True)


class MediaStorageSchema(BaseModel):
    media_id: int
    file_hash: str
    file_path: str
    created_at: datetime
    references: int
    description: str
    vectorized: bool
    blocked: bool


class ChatHistory(Model):
    msg_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(index=True)
    user_id: Mapped[str] = mapped_column(index=True)
    content_type: Mapped[str]
    content: Mapped[str]
    created_at: Mapped[datetime] = mapped_column(default=datetime.now, index=True)
    user_name: Mapped[str]
    media_id: Mapped[int | None]  # 媒体消息专用
    vectorized: Mapped[bool] = mapped_column(Boolean, default=False, index=True)


class UserRelation(Model):
    """用户关系/好感度表"""

    __tablename__ = "nonebot_plugin_ai_groupmate_userrelation_v2"
    __table_args__ = (UniqueConstraint("user_id", name="uq_nonebot_plugin_ai_groupmate_userrelation_v2_user_id"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(index=True)
    user_name: Mapped[str]
    favorability: Mapped[int] = mapped_column(default=0, index=True)  # 兼容旧逻辑的显示分
    favorability_raw: Mapped[int] = mapped_column(default=0, index=True)  # Monika风格原始分
    state: Mapped[str] = mapped_column(String(32), default="normal", index=True)
    tags: Mapped[list[str]] = mapped_column(JSON, default=list)
    daily_gain_used: Mapped[float] = mapped_column(Float, default=0.0)
    daily_bypass_used: Mapped[float] = mapped_column(Float, default=0.0)
    daily_gain_bank: Mapped[float] = mapped_column(Float, default=0.0)
    daily_cap: Mapped[float] = mapped_column(Float, default=7.0)
    cap_reset_at: Mapped[datetime] = mapped_column(default=datetime.now, index=True)
    last_interact_at: Mapped[datetime] = mapped_column(default=datetime.now, index=True)
    last_penalty_at: Mapped[datetime | None] = mapped_column(nullable=True)
    apology_counts: Mapped[dict[str, int]] = mapped_column(JSON, default=dict)
    updated_at: Mapped[datetime] = mapped_column(default=datetime.now, onupdate=datetime.now)

    def get_status_desc(self) -> str:
        """根据分数返回关系描述"""
        return status_desc_from_raw(self.favorability_raw)


class ChatHistorySchema(BaseModel):
    msg_id: int
    session_id: str
    user_id: str
    content_type: str
    content: str
    created_at: datetime
    user_name: str
    media_id: int | None = None
    vectorized: bool | None = False

    class Config:
        from_attributes = True  # ✅ 允许从 ORM 对象创建
