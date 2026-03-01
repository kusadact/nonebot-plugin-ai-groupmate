"""add blocked flag to media storage

迁移 ID: b3c9d8a7f102
父迁移: 6a6a44d58ced
创建时间: 2026-02-28 19:15:00.000000

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "b3c9d8a7f102"
down_revision: str | Sequence[str] | None = "6a6a44d58ced"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade(name: str = "") -> None:
    if name:
        return
    with op.batch_alter_table("nonebot_plugin_ai_groupmate_mediastorage", schema=None) as batch_op:
        batch_op.add_column(sa.Column("blocked", sa.Boolean(), nullable=False, server_default=sa.false()))
        batch_op.create_index(batch_op.f("ix_nonebot_plugin_ai_groupmate_mediastorage_blocked"), ["blocked"], unique=False)


def downgrade(name: str = "") -> None:
    if name:
        return
    with op.batch_alter_table("nonebot_plugin_ai_groupmate_mediastorage", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_nonebot_plugin_ai_groupmate_mediastorage_blocked"))
        batch_op.drop_column("blocked")
