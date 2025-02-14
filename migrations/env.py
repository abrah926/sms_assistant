import os
import sys
from logging.config import fileConfig
from dotenv import load_dotenv
from urllib.parse import quote_plus
from sqlalchemy import engine_from_config
from sqlalchemy import pool
from alembic import context

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models import Base

# Load environment variables from .env file
load_dotenv()

# this is the Alembic Config object
config = context.config

# Set up the database URL
def get_url():
    password = quote_plus(os.getenv('PGPASSWORD', ''))
    sslmode = os.getenv('PGSSLMODE', 'verify-ca')
    sslrootcert = os.path.expanduser(os.getenv('PGSSLROOTCERT', '~/.postgresql/root.crt'))
    
    return (f"postgresql://{os.getenv('PGUSER')}:{password}@"
            f"{os.getenv('PGHOST')}:{os.getenv('PGPORT', '5432')}/{os.getenv('PGDATABASE')}"
            f"?sslmode={sslmode}&sslrootcert={sslrootcert}")

# Override the SQLAlchemy URL with our constructed one
config.set_main_option("sqlalchemy.url", get_url())

# Interpret the config file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

def run_migrations_offline() -> None:
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online() -> None:
    configuration = config.get_section(config.config_ini_section)
    configuration["sqlalchemy.url"] = get_url()
    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, 
            target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online() 