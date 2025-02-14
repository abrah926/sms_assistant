"""add timescaledb extension

Revision ID: 2024_02_11_002
Revises: 2024_02_11_001
Create Date: 2024-02-11

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '2024_02_11_002'
down_revision = '2024_02_11_001'
branch_labels = None
depends_on = None

def upgrade():
    # Enable TimescaleDB extension
    op.execute('CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE')
    
    # Create messages table if it doesn't exist
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
    
    # Convert to hypertable
    op.execute("""
        SELECT create_hypertable('messages', 'timestamp', 
            if_not_exists => TRUE,
            chunk_time_interval => INTERVAL '1 day'
        )
    """)

def downgrade():
    op.drop_table('messages')
    op.execute('DROP EXTENSION IF EXISTS timescaledb CASCADE') 