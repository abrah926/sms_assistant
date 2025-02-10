from alembic import op
import sqlalchemy as sa

def upgrade():
    # Enable TimescaleDB extension
    op.execute('CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE')
    
    # Create hypertable
    op.execute("""
        SELECT create_hypertable('messages', 'timestamp', 
            if_not_exists => TRUE,
            chunk_time_interval => INTERVAL '1 day'
        )
    """)

def downgrade():
    op.execute('DROP EXTENSION timescaledb CASCADE') 