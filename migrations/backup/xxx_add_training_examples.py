"""add training examples table

Revision ID: 2024_02_11_001
Revises: 859fa4fe5bfe
Create Date: 2024-02-11

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision = '2024_02_11_001'
down_revision = '859fa4fe5bfe'
branch_labels = None
depends_on = None

def upgrade():
    op.create_table(
        'training_examples',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('customer_message', sa.String(), nullable=False),
        sa.Column('agent_response', sa.String(), nullable=False),
        sa.Column('context', sa.String()),
        sa.Column('meta_info', JSONB),  # Changed to match models.py
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_training_created', 'training_examples', ['created_at'])

def downgrade():
    op.drop_index('idx_training_created')
    op.drop_table('training_examples') 