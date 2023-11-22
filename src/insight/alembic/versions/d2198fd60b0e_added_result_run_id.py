"""Added Result.run_id

Revision ID: d2198fd60b0e
Revises: 9aca5ae68ff5
Create Date: 2023-09-14 12:25:17.878689

"""
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "d2198fd60b0e"
down_revision = "9aca5ae68ff5"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column("result", sa.Column("run_id", sa.VARCHAR(length=50), nullable=True, default=None))
    op.alter_column("version", "name", existing_type=sa.VARCHAR(length=50), nullable=False)
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.alter_column("version", "name", existing_type=sa.VARCHAR(length=50), nullable=True)
    op.drop_column("result", "run_id")
    # ### end Alembic commands ###