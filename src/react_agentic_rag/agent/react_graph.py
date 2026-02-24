"""ReAct 编排层。

优先使用 LangGraph 的预制 ReAct Agent；不可用时退化为本地可运行的最小实现。
"""

from __future__ import annotations

from typing import Any

from react_agentic_rag.rag.retriever import format_retrieval_output

SYSTEM_PROMPT = (
    "你是本地知识问答助手。"
    "回答必须基于工具返回证据，并在结论后列出引用 source 和 chunk。"
    "如果证据不足，直接说明不知道。"
)


class _FallbackAgent:
    """LangGraph 不可用时的兼容 Agent。

    该实现保持“先检索再回答”的行为，并提供与主流程兼容的 `invoke/stream` 接口。
    """

    def __init__(self, llm: Any, tool_func):
        """初始化 fallback agent。"""
        self._llm = llm
        self._tool_func = tool_func

    def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        """执行单轮问答并返回标准消息结构。"""
        messages = payload.get("messages", [])
        question = ""
        if messages:
            last = messages[-1]
            if isinstance(last, dict):
                question = str(last.get("content", ""))
            elif isinstance(last, tuple) and len(last) >= 2:
                question = str(last[1])
            else:
                question = str(last)

        context = self._tool_func(question)
        prompt = (
            f"{SYSTEM_PROMPT}\n\n用户问题:\n{question}\n\n"
            f"检索证据:\n{context}\n\n请给出回答。"
        )
        response = self._llm.invoke([{"role": "user", "content": prompt}])

        if isinstance(response, dict):
            messages = response.get("messages") if isinstance(response.get("messages"), list) else []
            if messages:
                last = messages[-1]
                if isinstance(last, dict):
                    final_content = str(last.get("content", ""))
                else:
                    final_content = str(last)
            else:
                final_content = str(response.get("content", ""))
        else:
            final_content = str(response)

        return {
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "name": "retrieve_context",
                            "args": {"query": question},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "name": "retrieve_context",
                    "content": context,
                },
                {
                    "role": "assistant",
                    "content": final_content,
                },
            ]
        }

    def stream(self, payload: dict[str, Any], stream_mode=None):
        """输出兼容 LangGraph 的流式事件。"""
        result = self.invoke(payload)
        messages = result.get("messages", [])

        if not isinstance(stream_mode, list):
            yield {"messages": messages}
            return

        emit_updates = "updates" in stream_mode
        emit_messages = "messages" in stream_mode

        for msg in messages:
            role = str(msg.get("role", "")).lower() if isinstance(msg, dict) else ""
            if role == "assistant" and isinstance(msg, dict) and msg.get("tool_calls"):
                if emit_updates:
                    yield ("updates", {"agent": {"messages": [msg]}})
                continue

            if role == "tool":
                if emit_updates:
                    yield ("updates", {"tools": {"messages": [msg]}})
                continue

            if role == "assistant":
                text = ""
                if isinstance(msg, dict):
                    text = str(msg.get("content", ""))
                else:
                    text = str(msg)
                if emit_messages:
                    for token in text:
                        yield ("messages", ({"role": "assistant", "content": token}, {"langgraph_node": "agent"}))
                elif emit_updates:
                    yield ("updates", {"agent": {"messages": [msg]}})


def build_react_agent(llm: Any, retriever: Any, top_k: int):
    """构建 ReAct Agent。

    Args:
        llm: 聊天模型实例（如 ChatOllama）。
        retriever: 具备 `search(query, top_k)` 方法的检索器。
        top_k: 每次检索的片段数。

    Returns:
        支持 `invoke`，并尽量支持 `stream` 的 Agent 实例。
    """

    def retrieve_context(query: str) -> str:
        """工具函数：执行检索并格式化证据文本。"""
        chunks = retriever.search(query, top_k=top_k)
        return format_retrieval_output(chunks)

    try:
        from langchain_core.tools import tool
        from langgraph.prebuilt import create_react_agent

        @tool("retrieve_context")
        def retrieve_context_tool(query: str) -> str:
            """检索本地知识库并返回带来源的文本片段。"""
            return retrieve_context(query)

        return create_react_agent(
            model=llm,
            tools=[retrieve_context_tool],
            prompt=SYSTEM_PROMPT,
        )
    except Exception:
        return _FallbackAgent(llm=llm, tool_func=retrieve_context)
