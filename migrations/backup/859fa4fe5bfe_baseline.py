"""baseline

Revision ID: 859fa4fe5bfe
Revises: 
Create Date: 2025-02-10 22:43:59.465971

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '859fa4fe5bfe'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create base tables
    op.create_table(
        'customers',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('phone', sa.String(), nullable=False),
        sa.Column('name', sa.String()),
        sa.Column('payment_info', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('phone')
    )

    op.create_table(
        'products',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('description', sa.Text()),
        sa.Column('price_per_kg', sa.Float(), nullable=False),
        sa.Column('min_order_kg', sa.Float(), nullable=False),
        sa.Column('available_kg', sa.Float(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )


def downgrade() -> None:
    op.drop_table('products')
    op.drop_table('customers') 