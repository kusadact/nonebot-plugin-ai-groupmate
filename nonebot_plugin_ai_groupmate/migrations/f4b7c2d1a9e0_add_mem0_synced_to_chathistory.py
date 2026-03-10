"""add mem0 synced flag to chat history

迁移 ID: f4b7c2d1a9e0
父迁移: e1a6f4d3c2b8
创建时间: 2026-03-10 10:15:00.000000

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "f4b7c2d1a9e0"
down_revision: str | Sequence[str] | None = "e1a6f4d3c2b8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TABLE = "nonebot_plugin_ai_groupmate_chathistory"


def upgrade(name: str = "") -> None:
    if name:
        return

    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())
    if TABLE not in tables:
        return

    columns = {col["name"] for col in inspector.get_columns(TABLE)}
    if "mem0_synced" in columns:
        return

    with op.batch_alter_table(TABLE, schema=None) as batch_op:
        batch_op.add_column(sa.Column("mem0_synced", sa.Boolean(), nullable=False, server_default=sa.false()))
        batch_op.create_index(batch_op.f("ix_nonebot_plugin_ai_groupmate_chathistory_mem0_synced"), ["mem0_synced"], unique=False)


def downgrade(name: str = "") -> None:
    if name:
        return

    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())
    if TABLE not in tables:
        return

    columns = {col["name"] for col in inspector.get_columns(TABLE)}
    if "mem0_synced" not in columns:
        return

    with op.batch_alter_table(TABLE, schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_nonebot_plugin_ai_groupmate_chathistory_mem0_synced"))
        batch_op.drop_column("mem0_synced")
