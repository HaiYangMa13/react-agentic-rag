"""面向业务层的结构化数据模型。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class SourceCitation:
    """单条证据引用。"""

    source: str
    chunk_id: str
    score: float


@dataclass(slots=True)
class AgentAnswer:
    """Agent 最终回答及其引用集合。"""

    answer: str
    citations: list[SourceCitation]
