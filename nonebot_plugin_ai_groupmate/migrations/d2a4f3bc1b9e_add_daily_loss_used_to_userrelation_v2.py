"""add daily_loss_used to userrelation_v2

迁移 ID: d2a4f3bc1b9e
父迁移: c7f99f9f4e21
创建时间: 2026-03-03 10:40:00.000000

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "d2a4f3bc1b9e"
down_revision: str | Sequence[str] | None = "c7f99f9f4e21"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TABLE = "nonebot_plugin_ai_groupmate_userrelation_v2"
COLUMN = "daily_loss_used"


def upgrade(name: str = "") -> None:
    if name:
        return

    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())
    if TABLE not in tables:
        return

    columns = {col["name"] for col in inspector.get_columns(TABLE)}
    if COLUMN in columns:
        return

    with op.batch_alter_table(TABLE, schema=None) as batch_op:
        batch_op.add_column(sa.Column(COLUMN, sa.Float(), nullable=False, server_default="0"))


def downgrade(name: str = "") -> None:
    if name:
        return

    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())
    if TABLE not in tables:
        return

    columns = {col["name"] for col in inspector.get_columns(TABLE)}
    if COLUMN not in columns:
        return

    with op.batch_alter_table(TABLE, schema=None) as batch_op:
        batch_op.drop_column(COLUMN)

