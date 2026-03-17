"""Database initialisation script for RetentionIQ.

Creates the medallion-layer schemas (bronze, silver, gold) and the
agent-memory table with pgvector support. Connection parameters are
read from the ``DATABASE_URL`` environment variable (or ``.env``
file via python-dotenv).

Usage:
    python scripts/init_db.py
"""

import os
import sys

import structlog
from dotenv import load_dotenv
from sqlalchemy import Engine, create_engine, text

logger = structlog.get_logger()


def get_database_url() -> str:
    """Return the DATABASE_URL from the environment.

    Loads ``.env`` first so local development works without exporting
    the variable manually.

    Returns:
        The database connection string.

    Raises:
        SystemExit: If ``DATABASE_URL`` is not set.
    """
    load_dotenv()
    url = os.environ.get("DATABASE_URL")
    if not url:
        logger.error(
            "DATABASE_URL not set. "
            "Please set it in the environment or in a .env file."
        )
        sys.exit(1)
    return url


def create_schemas(engine: Engine) -> None:
    """Create the bronze, silver, and gold schemas if they don't exist.

    Args:
        engine: SQLAlchemy engine instance.
    """
    schemas = ["bronze", "silver", "gold"]
    with engine.connect() as conn:
        for schema in schemas:
            conn.execute(
                text(f"CREATE SCHEMA IF NOT EXISTS {schema}")
            )
            logger.info("schema_created", schema=schema)
        conn.commit()


def enable_pgvector(engine: Engine) -> None:
    """Enable the pgvector extension.

    Args:
        engine: SQLAlchemy engine instance.
    """
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    logger.info("pgvector_extension_enabled")


def create_agent_memory_table(engine: Engine) -> None:
    """Create the ``agent_memory`` table with a pgvector column.

    Schema:
        * ``id`` -- UUID primary key (text for portability).
        * ``content`` -- The raw text content.
        * ``embedding`` -- 1536-dimensional vector (OpenAI ada-002 /
          text-embedding-3-small compatible).
        * ``metadata`` -- Arbitrary JSONB metadata.
        * ``created_at`` -- Timestamp defaulting to ``now()``.

    Args:
        engine: SQLAlchemy engine instance.
    """
    with engine.connect() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS agent_memory (
                    id          TEXT PRIMARY KEY,
                    content     TEXT NOT NULL,
                    embedding   vector(1536),
                    metadata    JSONB DEFAULT '{}'::jsonb,
                    created_at  TIMESTAMP WITH TIME ZONE
                                DEFAULT now()
                )
                """
            )
        )
        # Create an HNSW index for fast approximate nearest-neighbour
        # searches on the embedding column.
        conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_agent_memory_embedding
                ON agent_memory
                USING hnsw (embedding vector_cosine_ops)
                """
            )
        )
        conn.commit()
    logger.info("agent_memory_table_created")


def init_db() -> None:
    """Run the full database initialisation sequence."""
    url = get_database_url()
    logger.info("connecting_to_database", url=url.split("@")[-1])

    engine = create_engine(url)

    create_schemas(engine)
    enable_pgvector(engine)
    create_agent_memory_table(engine)

    engine.dispose()
    logger.info("database_initialisation_complete")


if __name__ == "__main__":
    init_db()
