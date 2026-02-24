"""检索层。

该模块屏蔽底层 FAISS 细节，对外提供稳定的检索结果结构。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class RetrievedChunk:
    """单条检索命中结果。"""

    text: str
    source: str
    chunk_id: str
    score: float


def format_retrieval_output(chunks: list[RetrievedChunk]) -> str:
    """把检索结果拼成可直接喂给 LLM 的证据文本。"""
    if not chunks:
        return "未检索到可用上下文。"

    lines = []
    for idx, chunk in enumerate(chunks, start=1):
        lines.append(
            f"[{idx}] source={chunk.source} chunk={chunk.chunk_id} score={chunk.score:.4f}\n{chunk.text}"
        )
    return "\n\n".join(lines)


class FaissRetriever:
    """FAISS 检索器包装层。"""

    def __init__(self, vector_store, top_k: int):
        """初始化检索器。

        Args:
            vector_store: 已加载的底层向量库实例。
            top_k: 默认召回数量。
        """
        self._vector_store = vector_store
        self._top_k = top_k

    @classmethod
    def from_disk(
        cls,
        vector_store_dir: Path,
        embedding_model: str,
        top_k: int,
    ) -> "FaissRetriever":
        """从磁盘加载 FAISS 索引并构造检索器。"""
        try:
            from langchain_community.vectorstores import FAISS
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError as exc:
            raise RuntimeError("缺少向量检索依赖，请先安装 project dependencies") from exc

        embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        db = FAISS.load_local(
            str(vector_store_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        return cls(db, top_k=top_k)

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """执行向量检索并映射为项目内部结构。"""
        k = top_k or self._top_k
        rows = self._vector_store.similarity_search_with_score(query, k=k)
        chunks: list[RetrievedChunk] = []
        for doc, score in rows:
            meta = doc.metadata or {}
            chunks.append(
                RetrievedChunk(
                    text=doc.page_content,
                    source=str(meta.get("source", "unknown")),
                    chunk_id=str(meta.get("chunk_id", "unknown")),
                    score=float(score),
                )
            )
        return chunks
