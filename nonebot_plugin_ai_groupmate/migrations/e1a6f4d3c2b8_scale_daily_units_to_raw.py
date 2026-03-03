"""scale daily favorability units to raw

迁移 ID: e1a6f4d3c2b8
父迁移: d2a4f3bc1b9e
创建时间: 2026-03-03 11:00:00.000000

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "e1a6f4d3c2b8"
down_revision: str | Sequence[str] | None = "d2a4f3bc1b9e"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TABLE = "nonebot_plugin_ai_groupmate_userrelation_v2"


def upgrade(name: str = "") -> None:
    if name:
        return

    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())
    if TABLE not in tables:
        return

    columns = {col["name"] for col in inspector.get_columns(TABLE)}
    required = {"daily_gain_used", "daily_loss_used", "daily_bypass_used", "daily_gain_bank", "daily_cap"}
    if not required.issubset(columns):
        return

    # 旧版本按映射分存储（日上限=7）。这里统一放大到 raw 单位。
    op.execute(
        sa.text(
            f"""
            UPDATE {TABLE}
            SET
                daily_gain_used = daily_gain_used * 10,
                daily_loss_used = daily_loss_used * 10,
                daily_bypass_used = daily_bypass_used * 10,
                daily_gain_bank = daily_gain_bank * 10,
                daily_cap = daily_cap * 10
            """
        )
    )

    with op.batch_alter_table(TABLE, schema=None) as batch_op:
        batch_op.alter_column("daily_cap", server_default="70")


def downgrade(name: str = "") -> None:
    if name:
        return

    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())
    if TABLE not in tables:
        return

    columns = {col["name"] for col in inspector.get_columns(TABLE)}
    required = {"daily_gain_used", "daily_loss_used", "daily_bypass_used", "daily_gain_bank", "daily_cap"}
    if not required.issubset(columns):
        return

    op.execute(
        sa.text(
            f"""
            UPDATE {TABLE}
            SET
                daily_gain_used = daily_gain_used / 10,
                daily_loss_used = daily_loss_used / 10,
                daily_bypass_used = daily_bypass_used / 10,
                daily_gain_bank = daily_gain_bank / 10,
                daily_cap = daily_cap / 10
            """
        )
    )

    with op.batch_alter_table(TABLE, schema=None) as batch_op:
        batch_op.alter_column("daily_cap", server_default="7")

