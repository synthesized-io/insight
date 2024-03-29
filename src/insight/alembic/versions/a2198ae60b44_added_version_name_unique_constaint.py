"""Set version.name.unique = True

Revision ID: a2198ae60b44
Revises: d2198fd60b0e
Create Date: 2023-12-13 13:25:17.878689

"""
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "a2198ae60b44"
down_revision = "d2198fd60b0e"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.alter_column(
        "version",
        "name",
        existing_type=sa.VARCHAR(length=50),
        unique=True,
        existing_nullable=False,
    )
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.alter_column(
        "version",
        "name",
        existing_type=sa.VARCHAR(length=50),
        unique=False,
        existing_nullable=False,
    )
    # ### end Alembic commands ###
