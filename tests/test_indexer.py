from pathlib import Path

from react_agentic_rag.rag.indexer import chunk_text, discover_documents


def test_chunk_text_with_overlap():
    text = " ".join([f"tok{i}" for i in range(20)])
    chunks = chunk_text(text, chunk_size=8, chunk_overlap=2)

    assert len(chunks) == 3
    assert chunks[0].split()[-2:] == chunks[1].split()[:2]


def test_discover_documents_only_supported(tmp_path: Path):
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "b.md").write_text("# md", encoding="utf-8")
    (tmp_path / "c.pdf").write_text("fake", encoding="utf-8")
    (tmp_path / "d.png").write_text("img", encoding="utf-8")

    docs = discover_documents(tmp_path)
    assert [p.name for p in docs] == ["a.txt", "b.md", "c.pdf"]
