"""Runbook knowledge base backed by LlamaIndex, exposed as a LangChain tool."""

from __future__ import annotations

import os
from functools import lru_cache

from langchain_core.tools import tool
from llama_index.core import SimpleDirectoryReader, Settings, VectorStoreIndex
from llama_index.core.retrievers import BaseRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

RUNBOOKS_DIR = os.path.join(os.path.dirname(__file__), "runbooks")

# Local embedding model, no API calls/credits required for indexing or querying.
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")


@lru_cache(maxsize=1)
def _get_retriever() -> BaseRetriever:
    """Build (once) and cache a retriever over the runbooks.

    Retrieval-only (no LLM-synthesized answer): the retrieved passages are
    fed as context into the RCA synthesizer's own LLM call, so a second
    LLM round-trip here would be redundant.

    Raises:
        FileNotFoundError: If the runbooks directory is missing or empty.
    """
    if not os.path.isdir(RUNBOOKS_DIR) or not os.listdir(RUNBOOKS_DIR):
        raise FileNotFoundError(
            f"No runbooks found in {RUNBOOKS_DIR!r}; cannot build knowledge base."
        )

    documents = SimpleDirectoryReader(RUNBOOKS_DIR).load_data()
    index = VectorStoreIndex.from_documents(documents)
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
