"""add training examples table

Revision ID: xxx
Revises: previous_revision
Create Date: 2024-02-11

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

def upgrade():
    op.create_table(
        'training_examples',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('customer_message', sa.String(), nullable=False),
        sa.Column('agent_response', sa.String(), nullable=False),
        sa.Column('context', sa.String()),
        sa.Column('metadata', JSONB),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_training_created', 'training_examples', ['created_at'])

def downgrade():
    op.drop_index('idx_training_created')
    op.drop_table('training_examples') 