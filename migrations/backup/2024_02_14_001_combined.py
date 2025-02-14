"""combined migration

Revision ID: 2024_02_14_001
Revises: 
Create Date: 2024-02-14

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision = '2024_02_14_001'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    # Enable TimescaleDB extension
    op.execute('CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE')
    
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

    op.create_table(
        'messages',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('phone', sa.String(), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('direction', sa.String(), nullable=False),
        sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False),
        sa.Column('meta_data', sa.JSON()),
        sa.Column('message_type', sa.String()),
        sa.PrimaryKeyConstraint('id')
    )

    # Convert messages to hypertable
    op.execute("""
        SELECT create_hypertable('messages', 'timestamp', 
            if_not_exists => TRUE,
            chunk_time_interval => INTERVAL '1 day'
        )
    """)

    op.create_table(
        'training_examples',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('customer_message', sa.String(), nullable=False),
        sa.Column('agent_response', sa.String(), nullable=False),
        sa.Column('context', sa.String()),
        sa.Column('meta_info', JSONB),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_training_created', 'training_examples', ['created_at'])

def downgrade():
    op.drop_index('idx_training_created')
    op.drop_table('training_examples')
    op.drop_table('messages')
    op.drop_table('products')
    op.drop_table('customers')
    op.execute('DROP EXTENSION IF EXISTS timescaledb CASCADE')