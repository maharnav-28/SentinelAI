"""Tests for RAG pipeline: chunking, FAISS vector retrieval, and zero-API-key fallback."""

import pytest
from pathlib import Path

from src.rag import (
    chunk_document,
    build_faiss_index,
    load_rag_artifacts,
    retrieve_relevant_chunks,
    answer_maintenance_question,
)
from src.utils import MAINTENANCE_DOCS_DIR


def test_chunk_document():
    """Verify markdown documents are chunked by section while retaining metadata."""
    sample_doc = MAINTENANCE_DOCS_DIR / "SOP_01_Thermal_Dissipation_HDF.md"
    assert sample_doc.exists()

    chunks = chunk_document(sample_doc)
    assert len(chunks) > 0

    first_chunk = chunks[0]
    assert "chunk_id" in first_chunk
    assert "doc_name" in first_chunk
    assert first_chunk["doc_name"] == "SOP_01_Thermal_Dissipation_HDF.md"
    assert "section" in first_chunk
    assert "text" in first_chunk
    assert len(first_chunk["text"]) > 20


def test_build_faiss_index_missing_dir(tmp_path: Path):
    """Verify build_faiss_index raises FileNotFoundError if no docs are present."""
    empty_dir = tmp_path / "empty_docs"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        build_faiss_index(empty_dir)


def test_rag_artifacts_load():
    """Verify saved FAISS index and chunk metadata load properly."""
    index, chunks, embedder = load_rag_artifacts()
    assert index.ntotal > 0
    assert len(chunks) == index.ntotal
    assert embedder is not None


def test_retrieve_relevant_chunks():
    """Verify FAISS vector retrieval returns semantically relevant chunks."""
    index, chunks, embedder = load_rag_artifacts()

    # Query about tool wear should retrieve tool wear document as top result
    query = "What is the tool wear limit and when should I replace the cutting insert?"
    results = retrieve_relevant_chunks(query, index, chunks, embedder, top_k=3)

    assert len(results) == 3
    top_result = results[0]
    assert "SOP_02_Tool_Wear_TWF.md" in top_result["doc_name"]
    assert top_result["similarity_score"] > 0.40


def test_retrieve_empty_query():
    """Verify empty or whitespace query returns empty list without error."""
    index, chunks, embedder = load_rag_artifacts()
    results = retrieve_relevant_chunks("   ", index, chunks, embedder)
    assert results == []


def test_answer_maintenance_question_zero_api_key_fallback(monkeypatch):
    """Verify RAG answers gracefully in retrieval-only mode when no API key is set."""
    # Ensure no API key is present in environment
    monkeypatch.setenv("LLM_API_KEY", "")

    index, chunks, embedder = load_rag_artifacts()
    query = "How often should spindle bearings be lubricated?"
    res = answer_maintenance_question(query, index, chunks, embedder, top_k=2)

    assert "mode" in res
    assert res["mode"] == "retrieval_only_fallback"
    assert "answer" in res
    assert "[INFO] Retrieval-Only Mode Active" in res["answer"]
    assert len(res["sources"]) == 2
    assert "SOP_04_Rotational_Speed_and_Bearings.md" in res["sources"][0]["doc_name"]
