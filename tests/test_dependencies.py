from pathlib import Path


def test_sentence_transformers_declared_in_dependencies():
    content = Path("pyproject.toml").read_text(encoding="utf-8")
    assert "sentence-transformers" in content
