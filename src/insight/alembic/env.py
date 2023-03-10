import os

from alembic import context
from sqlalchemy import engine_from_config, pool

from insight.database.schema import Base

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    if url is None:
        raise ValueError("No sqlalchemy.url specified in config file")

    url = url.format(**os.environ)
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """

    POSTGRES_PASSWORD = os.environ["POSTGRES_PASSWORD"]
    POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "localhost")
    POSTGRES_PORT = os.environ.get("POSTGRES_PORT", "5432")
    POSTGRES_USER = os.environ.get("POSTGRES_USER", "postgres")
    POSTGRES_DATABASE = os.environ.get("POSTGRES_DATABASE", "postgres")

    url = config.get_main_option("sqlalchemy.url")
    if url is None:
        raise ValueError("No sqlalchemy.url specified in config file")

    config.set_main_option("sqlalchemy.url", url.format(
        POSTGRES_USER=POSTGRES_USER,
        POSTGRES_PASSWORD=POSTGRES_PASSWORD,
        POSTGRES_HOST=POSTGRES_HOST,
        POSTGRES_PORT=POSTGRES_PORT,
        POSTGRES_DATABASE=POSTGRES_DATABASE
    ))
    connectable = engine_from_config(
        config.get_section(config.config_ini_section) or {},
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
