from pathlib import Path

from src.rag_pipeline import chunk_text, load_documents


def test_chunk_text_returns_overlapping_chunks():
    text = " ".join(f"word{i}" for i in range(20))
    chunks = chunk_text(text, chunk_size=8, overlap=2)

    assert len(chunks) == 3
    assert chunks[0].split()[-2:] == chunks[1].split()[:2]


def test_chunk_text_rejects_invalid_overlap():
    try:
        chunk_text("hello world", chunk_size=5, overlap=5)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")


def test_load_documents_reads_evaluation_corpus():
    docs = load_documents(Path("data"))

    assert docs
    names = {name for name, _ in docs}
    assert "healthcare_ai.txt" in names
