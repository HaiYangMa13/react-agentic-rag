"""索引构建层。

职责是“发现文档 -> 切块 -> 向量化 -> 持久化 FAISS 索引”。
"""

from __future__ import annotations

from pathlib import Path

from react_agentic_rag.rag.loader import load_document

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


def discover_documents(docs_dir: Path) -> list[Path]:
    """扫描目录并筛选支持的文档类型。"""
    docs = [
        p
        for p in sorted(docs_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return docs


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """按空白分词进行滑窗切块。

    该实现故意保持简单，便于学习数据流，后续可替换为更好的中文分段或语义切块。
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap 必须在 [0, chunk_size) 范围")

    tokens = text.split()
    if not tokens:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunks.append(" ".join(tokens[start:end]))
        if end == len(tokens):
            break
        # 通过回退 `chunk_overlap` 形成相邻片段重叠，降低跨块信息断裂。
        start = end - chunk_overlap
    return chunks


def build_faiss_index(
    docs_dir: Path,
    vector_store_dir: Path,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
) -> int:
    """构建并保存本地 FAISS 索引。

    Args:
        docs_dir: 原始文档目录。
        vector_store_dir: 索引输出目录。
        embedding_model: 嵌入模型名。
        chunk_size: 切块长度。
        chunk_overlap: 切块重叠长度。

    Returns:
        写入索引的总切块数量。
    """
    files = discover_documents(docs_dir)
    if not files:
        raise ValueError(f"目录中未发现可用文档: {docs_dir}")

    try:
        from langchain_community.vectorstores import FAISS
        from langchain_core.documents import Document
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError as exc:
        raise RuntimeError("缺少向量检索依赖，请先安装 project dependencies") from exc

    docs: list[Document] = []
    for file in files:
        loaded = load_document(file)
        chunks = chunk_text(
            loaded.text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        for idx, chunk in enumerate(chunks):
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": loaded.source,
                        "chunk_id": f"{Path(loaded.source).name}-{idx}",
                    },
                )
            )

    if not docs:
        raise ValueError("文档解析完成，但未生成有效切块")

    vector_store_dir.mkdir(parents=True, exist_ok=True)
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(str(vector_store_dir))
    return len(docs)
