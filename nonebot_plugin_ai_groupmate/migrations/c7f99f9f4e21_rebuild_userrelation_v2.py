"""rebuild userrelation to v2 schema

迁移 ID: c7f99f9f4e21
父迁移: b3c9d8a7f102
创建时间: 2026-03-03 09:20:00.000000

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c7f99f9f4e21"
down_revision: str | Sequence[str] | None = "b3c9d8a7f102"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

OLD_TABLE = "nonebot_plugin_ai_groupmate_userrelation"
NEW_TABLE = "nonebot_plugin_ai_groupmate_userrelation_v2"


def upgrade(name: str = "") -> None:
    if name:
        return

    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing = set(inspector.get_table_names())

    if NEW_TABLE in existing:
        return

    op.create_table(
        NEW_TABLE,
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("user_name", sa.String(), nullable=False),
        sa.Column("favorability", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("favorability_raw", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("state", sa.String(length=32), nullable=False, server_default="normal"),
        sa.Column("tags", sa.JSON(), nullable=False),
        sa.Column("daily_gain_used", sa.Float(), nullable=False, server_default="0"),
        sa.Column("daily_bypass_used", sa.Float(), nullable=False, server_default="0"),
        sa.Column("daily_gain_bank", sa.Float(), nullable=False, server_default="0"),
        sa.Column("daily_cap", sa.Float(), nullable=False, server_default="7"),
        sa.Column("cap_reset_at", sa.DateTime(), nullable=False),
        sa.Column("last_interact_at", sa.DateTime(), nullable=False),
        sa.Column("last_penalty_at", sa.DateTime(), nullable=True),
        sa.Column("apology_counts", sa.JSON(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_nonebot_plugin_ai_groupmate_userrelation_v2")),
        sa.UniqueConstraint("user_id", name=op.f("uq_nonebot_plugin_ai_groupmate_userrelation_v2_user_id")),
        info={"bind_key": "nonebot_plugin_ai_groupmate"},
    )

    with op.batch_alter_table(NEW_TABLE, schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_nonebot_plugin_ai_groupmate_userrelation_v2_cap_reset_at"), ["cap_reset_at"], unique=False)
        batch_op.create_index(batch_op.f("ix_nonebot_plugin_ai_groupmate_userrelation_v2_favorability"), ["favorability"], unique=False)
        batch_op.create_index(batch_op.f("ix_nonebot_plugin_ai_groupmate_userrelation_v2_favorability_raw"), ["favorability_raw"], unique=False)
        batch_op.create_index(batch_op.f("ix_nonebot_plugin_ai_groupmate_userrelation_v2_last_interact_at"), ["last_interact_at"], unique=False)
        batch_op.create_index(batch_op.f("ix_nonebot_plugin_ai_groupmate_userrelation_v2_state"), ["state"], unique=False)
        batch_op.create_index(batch_op.f("ix_nonebot_plugin_ai_groupmate_userrelation_v2_user_id"), ["user_id"], unique=False)


def downgrade(name: str = "") -> None:
    if name:
        return

    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing = set(inspector.get_table_names())

    if NEW_TABLE in existing:
        with op.batch_alter_table(NEW_TABLE, schema=None) as batch_op:
            batch_op.drop_index(batch_op.f("ix_nonebot_plugin_ai_groupmate_userrelation_v2_user_id"))
            batch_op.drop_index(batch_op.f("ix_nonebot_plugin_ai_groupmate_userrelation_v2_state"))
            batch_op.drop_index(batch_op.f("ix_nonebot_plugin_ai_groupmate_userrelation_v2_last_interact_at"))
            batch_op.drop_index(batch_op.f("ix_nonebot_plugin_ai_groupmate_userrelation_v2_favorability_raw"))
            batch_op.drop_index(batch_op.f("ix_nonebot_plugin_ai_groupmate_userrelation_v2_favorability"))
            batch_op.drop_index(batch_op.f("ix_nonebot_plugin_ai_groupmate_userrelation_v2_cap_reset_at"))

        op.drop_table(NEW_TABLE)
