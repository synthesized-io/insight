"""Add result sub_run

Revision ID: 495f69cbfe6d
Revises: a2198ae60b44
Create Date: 2024-03-05 16:01:49.471831

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "495f69cbfe6d"
down_revision = "a2198ae60b44"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column("result", sa.Column("sub_run", sa.INTEGER(), nullable=True))
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column("result", "sub_run")
    # ### end Alembic commands ###