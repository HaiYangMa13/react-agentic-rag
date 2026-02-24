from react_agentic_rag.config import Settings


def test_settings_defaults(tmp_path):
    settings = Settings.from_env(base_dir=tmp_path)
    assert settings.ollama_model == "qwen2.5:7b-instruct-q4_K_M"
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.rag_top_k == 4
    assert settings.vector_store_dir == tmp_path / "data" / "vectorstore"


def test_settings_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5:14b")
    monkeypatch.setenv("RAG_TOP_K", "6")

    settings = Settings.from_env(base_dir=tmp_path)

    assert settings.ollama_model == "qwen2.5:14b"
    assert settings.rag_top_k == 6
