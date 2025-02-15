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
    pg_user = os.getenv("PGUSER", "default_user")
    pg_password = os.getenv("PGPASSWORD", "default_password")
    pg_host = os.getenv("PGHOST", "localhost")
    pg_port = os.getenv("PGPORT", "5432")
    pg_database = os.getenv("PGDATABASE", "postgres")
    pg_sslmode = os.getenv("PGSSLMODE", "verify-ca")
    pg_sslrootcert = os.getenv("PGSSLROOTCERT", "/etc/ssl/certs/azure-postgres.crt")

    return f"postgresql://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_database}?sslmode={pg_sslmode}&sslrootcert={pg_sslrootcert}"

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