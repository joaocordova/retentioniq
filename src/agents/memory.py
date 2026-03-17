"""pgvector-based memory store for the agent layer.

This module provides short-term (conversation) and long-term (semantic)
memory for the LangGraph agent system. Memories are stored in PostgreSQL
with pgvector for efficient cosine similarity search.

Architecture:
    - Conversation memory: recent messages in a session (windowed)
    - Semantic memory: embeddings of past interactions, retrieved by similarity
    - Both types are stored in the same table, differentiated by memory_type

Security:
    - Connection strings come from environment variables, NEVER hardcoded
    - All stored content is PII-masked BEFORE storage
    - Sessions are clearable for LGPD compliance (right to deletion)
"""

import os
import uuid
from datetime import datetime, timezone
from typing import Any

import numpy as np
import structlog

from src.exceptions import AgentError

logger = structlog.get_logger()


class AgentMemoryStore:
    """pgvector-backed memory store for agent conversations and context.

    Stores conversation history and semantic memories with vector
    embeddings for similarity-based retrieval. Uses PostgreSQL with
    the pgvector extension.

    The store supports two memory types:
    - 'conversation': sequential messages within a session
    - 'semantic': reusable knowledge chunks retrieved by similarity

    Connection credentials are read from environment variables.
    """

    def __init__(
        self,
        connection_string: str | None = None,
        table_name: str = "agent_memory",
        embedding_dim: int = 1536,
    ) -> None:
        """Initialize the memory store.

        Connection string is read from the RETENTIONIQ_DB_URL environment
        variable if not provided directly. This ensures credentials are
        never hardcoded in source code.

        Args:
            connection_string: PostgreSQL connection string. If None,
                reads from RETENTIONIQ_DB_URL env var.
            table_name: Name of the memory table in PostgreSQL.
            embedding_dim: Dimension of embedding vectors (must match
                the embedding model — 1536 for text-embedding-3-small).

        Raises:
            AgentError: If no connection string is available.
        """
        if connection_string is None:
            connection_string = os.environ.get("RETENTIONIQ_DB_URL")
        if not connection_string:
            raise AgentError(
                "No database connection string provided. "
                "Set RETENTIONIQ_DB_URL environment variable."
            )

        self._connection_string = connection_string
        self._table_name = table_name
        self._embedding_dim = embedding_dim
        self._conn = None

        logger.info(
            "memory_store_initialized",
            table_name=table_name,
            embedding_dim=embedding_dim,
        )

    def _get_connection(self) -> Any:
        """Get or create a database connection.

        Returns:
            psycopg2 connection object.

        Raises:
            AgentError: If connection fails.
        """
        if self._conn is not None and not self._conn.closed:
            return self._conn

        try:
            import psycopg2

            self._conn = psycopg2.connect(self._connection_string)
            return self._conn
        except ImportError:
            raise AgentError(
                "psycopg2 is required for AgentMemoryStore. "
                "Install with: pip install psycopg2-binary"
            )
        except Exception as exc:
            raise AgentError(
                f"Failed to connect to database: {exc}"
            ) from exc

    def _ensure_table(self) -> None:
        """Create the memory table and pgvector extension if needed."""
        conn = self._get_connection()
        cur = conn.cursor()
        try:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._table_name} (
                    memory_id UUID PRIMARY KEY,
                    session_id VARCHAR(256),
                    content TEXT NOT NULL,
                    metadata JSONB DEFAULT '{{}}'::jsonb,
                    memory_type VARCHAR(50) NOT NULL
                        DEFAULT 'conversation',
                    embedding vector({self._embedding_dim}),
                    created_at TIMESTAMPTZ NOT NULL
                        DEFAULT NOW()
                );
            """)
            # Index for similarity search
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self._table_name}_embedding
                ON {self._table_name}
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            """)
            # Index for session lookups
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS
                    idx_{self._table_name}_session
                ON {self._table_name} (session_id, created_at);
            """)
            conn.commit()
            logger.info("memory_table_ensured", table=self._table_name)
        except Exception as exc:
            conn.rollback()
            raise AgentError(
                f"Failed to create memory table: {exc}"
            ) from exc
        finally:
            cur.close()

    def _generate_embedding(self, text: str) -> list[float]:
        """Generate an embedding vector for the given text.

        This is a placeholder that generates a deterministic
        pseudo-embedding based on text hashing. In production,
        replace with actual embedding API call (e.g., OpenAI
        text-embedding-3-small).

        Args:
            text: Text to embed.

        Returns:
            List of floats representing the embedding vector.
        """
        # Deterministic pseudo-embedding for development/testing.
        # In production, call the embedding API configured in
        # configs/agents.yaml (e.g., OpenAI text-embedding-3-small).
        rng = np.random.RandomState(
            hash(text) % (2**31)
        )
        vec = rng.randn(self._embedding_dim).astype(float)
        # Normalize to unit vector for cosine similarity
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def store_memory(
        self,
        content: str,
        metadata: dict[str, Any],
        memory_type: str = "conversation",
    ) -> str:
        """Store a memory with its embedding.

        Content should already be PII-masked before calling this method.
        The guardrails module handles masking upstream.

        Args:
            content: Text content to store (must be PII-masked).
            metadata: Arbitrary metadata dict (session_id, role, etc.).
                Stored as JSONB for flexible querying.
            memory_type: Type of memory — 'conversation' for chat
                messages, 'semantic' for reusable knowledge.

        Returns:
            UUID string of the stored memory.

        Raises:
            AgentError: If storage fails.
        """
        self._ensure_table()
        conn = self._get_connection()

        memory_id = str(uuid.uuid4())
        session_id = metadata.get("session_id", "")
        embedding = self._generate_embedding(content)

        import json

        cur = conn.cursor()
        try:
            cur.execute(
                f"""
                INSERT INTO {self._table_name}
                    (memory_id, session_id, content, metadata,
                     memory_type, embedding, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    memory_id,
                    session_id,
                    content,
                    json.dumps(metadata),
                    memory_type,
                    str(embedding),
                    datetime.now(timezone.utc),
                ),
            )
            conn.commit()
            logger.info(
                "memory_stored",
                memory_id=memory_id,
                memory_type=memory_type,
                session_id=session_id,
            )
            return memory_id
        except Exception as exc:
            conn.rollback()
            raise AgentError(
                f"Failed to store memory: {exc}"
            ) from exc
        finally:
            cur.close()

    def search_similar(
        self,
        query: str,
        top_k: int = 5,
        memory_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search for memories similar to the query using cosine similarity.

        Uses pgvector's cosine distance operator to find the most
        relevant past memories. Useful for retrieving context from
        previous interactions or similar questions.

        Args:
            query: Query text to find similar memories for.
            top_k: Maximum number of results to return.
            memory_type: Filter by memory type. None returns all types.

        Returns:
            List of dicts with keys: memory_id, content, metadata,
            memory_type, similarity, created_at. Sorted by similarity
            descending.

        Raises:
            AgentError: If search fails.
        """
        self._ensure_table()
        conn = self._get_connection()

        query_embedding = self._generate_embedding(query)
        cur = conn.cursor()
        try:
            if memory_type:
                cur.execute(
                    f"""
                    SELECT memory_id, content, metadata, memory_type,
                        1 - (embedding <=> %s::vector) AS similarity,
                        created_at
                    FROM {self._table_name}
                    WHERE memory_type = %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (
                        str(query_embedding),
                        memory_type,
                        str(query_embedding),
                        top_k,
                    ),
                )
            else:
                cur.execute(
                    f"""
                    SELECT memory_id, content, metadata, memory_type,
                        1 - (embedding <=> %s::vector) AS similarity,
                        created_at
                    FROM {self._table_name}
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (
                        str(query_embedding),
                        str(query_embedding),
                        top_k,
                    ),
                )

            rows = cur.fetchall()
            results = []
            for row in rows:
                import json

                results.append({
                    "memory_id": str(row[0]),
                    "content": row[1],
                    "metadata": (
                        json.loads(row[2])
                        if isinstance(row[2], str)
                        else row[2]
                    ),
                    "memory_type": row[3],
                    "similarity": float(row[4]),
                    "created_at": str(row[5]),
                })

            logger.info(
                "memory_search_complete",
                query_length=len(query),
                results_found=len(results),
            )
            return results

        except Exception as exc:
            raise AgentError(
                f"Memory search failed: {exc}"
            ) from exc
        finally:
            cur.close()

    def get_conversation_history(
        self,
        session_id: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get recent conversation messages for a session.

        Retrieves the most recent messages in chronological order.
        Used to maintain conversation context within the agent.

        Args:
            session_id: Session identifier to filter by.
            limit: Maximum number of messages to return.

        Returns:
            List of dicts with keys: memory_id, content, metadata,
            created_at. Ordered by created_at ascending (oldest first).

        Raises:
            AgentError: If retrieval fails.
        """
        self._ensure_table()
        conn = self._get_connection()
        cur = conn.cursor()
        try:
            cur.execute(
                f"""
                SELECT memory_id, content, metadata, created_at
                FROM {self._table_name}
                WHERE session_id = %s AND memory_type = 'conversation'
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (session_id, limit),
            )
            rows = cur.fetchall()

            import json

            results = []
            for row in rows:
                results.append({
                    "memory_id": str(row[0]),
                    "content": row[1],
                    "metadata": (
                        json.loads(row[2])
                        if isinstance(row[2], str)
                        else row[2]
                    ),
                    "created_at": str(row[3]),
                })

            # Return in chronological order (oldest first)
            results.reverse()

            logger.info(
                "conversation_history_retrieved",
                session_id=session_id,
                messages=len(results),
            )
            return results

        except Exception as exc:
            raise AgentError(
                f"Failed to retrieve conversation history: {exc}"
            ) from exc
        finally:
            cur.close()

    def clear_session(self, session_id: str) -> None:
        """Delete all memories for a session.

        Required for LGPD compliance — users have the right to request
        deletion of their conversation data. This permanently removes
        all memories associated with the session.

        Args:
            session_id: Session identifier whose memories to delete.

        Raises:
            AgentError: If deletion fails.
        """
        self._ensure_table()
        conn = self._get_connection()
        cur = conn.cursor()
        try:
            cur.execute(
                f"""
                DELETE FROM {self._table_name}
                WHERE session_id = %s
                """,
                (session_id,),
            )
            deleted = cur.rowcount
            conn.commit()
            logger.info(
                "session_cleared",
                session_id=session_id,
                memories_deleted=deleted,
            )
        except Exception as exc:
            conn.rollback()
            raise AgentError(
                f"Failed to clear session: {exc}"
            ) from exc
        finally:
            cur.close()
