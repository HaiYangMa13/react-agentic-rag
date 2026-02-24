"""项目运行时配置。

该模块把所有可配置项集中在 `Settings` 中，便于 CLI、API、测试共享同一套参数来源。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class Settings:
    """应用配置对象。

    Attributes:
        ollama_model: 对话模型名，交给 Ollama 调用。
        ollama_base_url: Ollama 服务地址。
        rag_top_k: 默认检索返回片段数。
        embedding_model: 嵌入模型名。
        docs_dir: 原始文档目录。
        vector_store_dir: 向量索引目录。
        chunk_size: 切块长度（按空白分词后的 token 数）。
        chunk_overlap: 切块重叠长度。
    """

    ollama_model: str
    ollama_base_url: str
    rag_top_k: int
    embedding_model: str
    docs_dir: Path
    vector_store_dir: Path
    chunk_size: int
    chunk_overlap: int

    @classmethod
    def from_env(cls, base_dir: Path | None = None) -> "Settings":
        """从环境变量与默认值构建配置。

        Args:
            base_dir: 项目根目录。为空时使用当前工作目录。

        Returns:
            完整的 `Settings` 实例。
        """
        root = (base_dir or Path.cwd()).resolve()
        return cls(
            ollama_model=os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct-q4_K_M"),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            rag_top_k=int(os.getenv("RAG_TOP_K", "4")),
            embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
            docs_dir=root / "data" / "docs",
            vector_store_dir=root / "data" / "vectorstore",
            chunk_size=int(os.getenv("CHUNK_SIZE", "400")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "80")),
        )
