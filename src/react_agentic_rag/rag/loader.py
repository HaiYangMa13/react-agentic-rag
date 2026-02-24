"""文档加载层。

当前支持 txt / md / pdf 三种输入，统一转换成纯文本结构，供后续切块和向量化使用。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class LoadedDocument:
    """统一的文档载体。"""

    source: str
    text: str


def load_document(path: Path) -> LoadedDocument:
    """读取单个文档并返回标准化文本。

    Args:
        path: 文档路径。

    Returns:
        `LoadedDocument`，其中 `text` 为可切块的纯文本。

    Raises:
        RuntimeError: 处理 PDF 但缺少 PyMuPDF 依赖。
        ValueError: 文件格式不受支持。
    """
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return LoadedDocument(source=str(path), text=path.read_text(encoding="utf-8"))
    if suffix == ".pdf":
        try:
            import fitz  # type: ignore
        except ImportError as exc:
            raise RuntimeError("缺少 PyMuPDF，请先安装依赖") from exc

        with fitz.open(path) as pdf:
            pages = [page.get_text("text") for page in pdf]
        return LoadedDocument(source=str(path), text="\n".join(pages))

    raise ValueError(f"不支持的文档格式: {path.suffix}")
