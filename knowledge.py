"""Runbook knowledge base backed by LlamaIndex + pgvector, exposed as a
LangChain tool.

Storage layer only: the embedding model (local HuggingFace, no external
API) and the runbooks/ source documents are unchanged from the original
in-memory LlamaIndex index. This swaps that in-memory index for a
Postgres + pgvector-backed one, so the index persists across process
restarts instead of being rebuilt from scratch every time (notably: every
MCP tool call, since mcp_server.py is re-spawned as a fresh subprocess per
call - see README.md's MCP section).
"""

from __future__ import annotations

import os
from functools import lru_cache

import psycopg2
from langchain_core.tools import tool
from llama_index.core import (
    SimpleDirectoryReader,
    Settings,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.retrievers import BaseRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.postgres import PGVectorStore

RUNBOOKS_DIR = os.path.join(os.path.dirname(__file__), "runbooks")

# Local embedding model, no API calls/credits required for indexing or querying.
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# bge-small-en-v1.5's output dimension - verified directly against the
# loaded model (`len(embed_model.get_text_embedding("test"))`), not assumed.
_EMBED_DIM = 384

# Local-only Postgres (see docker-compose.yml); overridable via env vars for
# anyone running Postgres with different settings.
_PG_HOST = os.environ.get("PGVECTOR_HOST", "localhost")
_PG_PORT = os.environ.get("PGVECTOR_PORT", "5432")
_PG_DATABASE = os.environ.get("PGVECTOR_DATABASE", "incident_triage")
_PG_USER = os.environ.get("PGVECTOR_USER", "incident_triage")
_PG_PASSWORD = os.environ.get("PGVECTOR_PASSWORD", "incident_triage")
_PG_TABLE_NAME = "runbooks"  # PGVectorStore creates/reads it as "data_runbooks"


def _runbooks_table_has_rows() -> bool:
    """Whether the pgvector table already has ingested runbook rows.

    Used to decide, on each process startup, whether to ingest runbooks/
    (first run) or just open the existing index (every run after). A
    separate explicit ingestion script was considered and rejected: it
    would be one more manual step to remember, and with the MCP server
    re-spawning as a fresh subprocess per tool call, "ingest if empty" is
    what makes ingestion happen exactly once ever, rather than needing to
    be triggered out of band.
    """
    conn = psycopg2.connect(
        host=_PG_HOST,
        port=_PG_PORT,
        dbname=_PG_DATABASE,
        user=_PG_USER,
        password=_PG_PASSWORD,
    )
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT EXISTS (SELECT FROM information_schema.tables "
                "WHERE table_name = %s)",
                (f"data_{_PG_TABLE_NAME}",),
            )
            if not cur.fetchone()[0]:
                return False
            cur.execute(f'SELECT COUNT(*) FROM "data_{_PG_TABLE_NAME}"')
            return cur.fetchone()[0] > 0
    finally:
        conn.close()


@lru_cache(maxsize=1)
def _get_retriever() -> BaseRetriever:
    """Build (once per process) and cache a retriever over the runbooks,
    backed by the pgvector-persisted index.

    Retrieval-only (no LLM-synthesized answer): the retrieved passages are
    fed as context into the RCA synthesizer's own LLM call, so a second
    LLM round-trip here would be redundant.

    Raises:
        FileNotFoundError: If the runbooks directory is missing or empty
            (only matters on first-ever ingestion; an already-populated
            table doesn't need runbooks/ to be present).
    """
    vector_store = PGVectorStore.from_params(
        host=_PG_HOST,
        port=_PG_PORT,
        database=_PG_DATABASE,
        user=_PG_USER,
        password=_PG_PASSWORD,
        table_name=_PG_TABLE_NAME,
        embed_dim=_EMBED_DIM,
        perform_setup=True,
    )

    if _runbooks_table_has_rows():
        index = VectorStoreIndex.from_vector_store(vector_store)
    else:
        if not os.path.isdir(RUNBOOKS_DIR) or not os.listdir(RUNBOOKS_DIR):
            raise FileNotFoundError(
                f"No runbooks found in {RUNBOOKS_DIR!r}; cannot ingest the "
                "knowledge base."
            )
        documents = SimpleDirectoryReader(RUNBOOKS_DIR).load_data()
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context
        )

    return index.as_retriever(similarity_top_k=3)


def query_runbooks(query: str) -> str:
    """Search the indexed runbooks for passages relevant to `query`.

    Args:
        query: A natural-language description of the incident/service,
            e.g. "payment-service 504 timeouts connection pool".

    Returns:
        The most relevant runbook passages, concatenated, or an
        explanatory message if the knowledge base could not be built.
    """
    try:
        retriever = _get_retriever()
    except FileNotFoundError as exc:
        return f"Runbook knowledge base unavailable: {exc}"
    except psycopg2.OperationalError as exc:
        return f"Runbook knowledge base unavailable (Postgres unreachable): {exc}"

    nodes = retriever.retrieve(query)
    if not nodes:
        return "No matching runbook guidance found."
    return "\n\n".join(node.get_content() for node in nodes)


@tool
def query_runbooks_tool(query: str) -> str:
    """Search internal SRE runbooks for incident diagnosis and remediation
    guidance relevant to the given query (service name + symptoms).

    Args:
        query: A natural-language description of the incident/service.

    Returns:
        Relevant runbook guidance as text.
    """
    return query_runbooks(query)
