"""命令行入口。

该模块提供两类能力：
1. `ingest`：把文档构建为本地 FAISS 索引。
2. `chat`：执行 ReAct 问答，并将关键阶段格式化输出到终端。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from react_agentic_rag.agent.react_graph import build_react_agent
from react_agentic_rag.config import Settings
from react_agentic_rag.rag.indexer import build_faiss_index
from react_agentic_rag.rag.retriever import FaissRetriever

RESET = "\033[0m"
THOUGHT_COLOR = "\033[35m"
ACTION_COLOR = "\033[36m"
OBSERVATION_COLOR = "\033[33m"
FINAL_COLOR = "\033[32m"
MAX_NON_FINAL_CHARS = 200


def build_parser() -> argparse.ArgumentParser:
    """构建 CLI 参数解析器。"""
    parser = argparse.ArgumentParser(prog="react-rag", description="Local ReAct + Agentic RAG scaffold")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="导入文档并构建向量索引")
    ingest.add_argument("--docs-dir", type=Path)
    ingest.add_argument("--vector-store-dir", type=Path)
    ingest.add_argument("--embedding-model", type=str)
    ingest.add_argument("--chunk-size", type=int)
    ingest.add_argument("--chunk-overlap", type=int)

    chat = sub.add_parser("chat", help="执行单轮问答")
    chat.add_argument("--question", type=str, required=True)
    chat.add_argument("--vector-store-dir", type=Path)
    chat.add_argument("--embedding-model", type=str)
    chat.add_argument("--top-k", type=int)
    chat.add_argument("--model", type=str)
    chat.add_argument("--ollama-base-url", type=str)

    return parser


def _extract_assistant_content(result) -> str:
    """从通用结果结构中提取最终回答文本。"""
    if isinstance(result, dict):
        messages = result.get("messages") or []
        if messages:
            msg = messages[-1]
            if isinstance(msg, dict):
                return str(msg.get("content", ""))
            return str(msg)
    return str(result)


def _msg_value(msg, key: str, default=None):
    """同时兼容 dict 消息与对象消息的字段读取。"""
    if isinstance(msg, dict):
        return msg.get(key, default)
    return getattr(msg, key, default)


def _msg_role(msg) -> str:
    """提取消息角色并统一为小写字符串。"""
    role = _msg_value(msg, "role")
    if role:
        return str(role).lower()
    msg_type = _msg_value(msg, "type")
    if msg_type:
        return str(msg_type).lower()
    return msg.__class__.__name__.lower()


def _content_to_text(content, strip: bool = True) -> str:
    """把不同 content 结构（str/list/object）统一转成文本。"""
    if isinstance(content, str):
        return content.strip() if strip else content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        merged = " ".join([p for p in parts if p])
        return merged.strip() if strip else merged
    if content is None:
        return ""
    text = str(content)
    return text.strip() if strip else text


def _tool_calls(msg) -> list:
    """获取消息里的工具调用列表。"""
    calls = _msg_value(msg, "tool_calls")
    if calls:
        return list(calls)
    additional_kwargs = _msg_value(msg, "additional_kwargs", {})
    if isinstance(additional_kwargs, dict):
        raw = additional_kwargs.get("tool_calls")
        if raw:
            return list(raw)
    return []


def _tool_call_chunks(msg) -> list:
    """获取流式消息里的工具调用分片。"""
    chunks = _msg_value(msg, "tool_call_chunks")
    if chunks:
        return list(chunks)
    return []


def _stage_prefix(stage: str) -> str:
    """生成带颜色的阶段前缀标签。"""
    if stage == "THOUGHT":
        return f"{THOUGHT_COLOR}[THOUGHT]{RESET}"
    if stage == "ACTION":
        return f"{ACTION_COLOR}[ACTION]{RESET}"
    if stage == "OBSERVATION":
        return f"{OBSERVATION_COLOR}[OBSERVATION]{RESET}"
    return f"{FINAL_COLOR}[FINAL]{RESET}"


def _truncate_non_final(text: str) -> str:
    """截断非最终阶段文本，避免 Observation 刷屏。"""
    if len(text) <= MAX_NON_FINAL_CHARS:
        return text
    omitted = len(text) - MAX_NON_FINAL_CHARS
    return f"{text[:MAX_NON_FINAL_CHARS]} [...省略{omitted}字]"


def _normalize_args(args):
    """归一化工具参数，便于去重与稳定打印。"""
    if isinstance(args, str):
        text = args.strip()
        if not text:
            return ""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text
    return args


def _format_args(args) -> str:
    """把归一化后的参数序列化为稳定字符串。"""
    normalized = _normalize_args(args)
    if isinstance(normalized, (dict, list)):
        return json.dumps(normalized, ensure_ascii=False, sort_keys=True)
    return str(normalized)


def _thought_summary(args) -> str:
    """基于工具参数生成可解释的思考摘要。"""
    normalized = _normalize_args(args)
    query = ""
    if isinstance(normalized, dict):
        query = str(normalized.get("query", "")).strip()
    elif isinstance(normalized, str):
        query = normalized.strip()

    if query:
        if len(query) > 32:
            query = f"{query[:32]}..."
        return f"{_stage_prefix('THOUGHT')} 先检索与“{query}”相关的证据，再给出结论。"
    return f"{_stage_prefix('THOUGHT')} 先检索相关证据，再给出结论。"


def _format_action(name: str, args) -> str:
    """格式化 Action 行。"""
    return f"{_stage_prefix('ACTION')} {name}({_format_args(args)})"


def _action_key(name: str, args) -> tuple[str, str]:
    """生成 Action 去重键。"""
    return name, _format_args(args)


def format_react_trace(result) -> str:
    """把一次完整推理结果格式化为可读轨迹。"""
    messages = []
    if isinstance(result, dict):
        raw_messages = result.get("messages")
        if isinstance(raw_messages, list):
            messages = raw_messages

    lines: list[str] = []
    seen_actions: set[tuple[str, str]] = set()
    final_answer = ""
    for msg in messages:
        role = _msg_role(msg)
        calls = _tool_calls(msg)
        if calls:
            for call in calls:
                if isinstance(call, dict):
                    name = str(call.get("name") or "tool")
                    args = call.get("args", call.get("arguments", {}))
                else:
                    name = str(getattr(call, "name", "tool"))
                    args = getattr(call, "args", getattr(call, "arguments", {}))
                key = _action_key(name, args)
                if key not in seen_actions:
                    lines.append(_thought_summary(args))
                    lines.append(_format_action(name, args))
                    seen_actions.add(key)

        if role in {"tool", "toolmessage"}:
            observation = _content_to_text(_msg_value(msg, "content", ""))
            lines.append(f"{_stage_prefix('OBSERVATION')} {_truncate_non_final(observation)}")
            continue

        if role in {"assistant", "ai", "aimessage"} and not calls:
            content = _content_to_text(_msg_value(msg, "content", ""))
            if content:
                final_answer = content

    if final_answer:
        lines.append(f"{_stage_prefix('FINAL')} {final_answer}")

    if not lines:
        return f"{_stage_prefix('FINAL')} {_extract_assistant_content(result)}"
    return "\n".join(lines)


def _iter_messages(payload):
    """深度遍历任意事件结构，提取其中的消息对象。"""
    if isinstance(payload, dict):
        messages = payload.get("messages")
        if isinstance(messages, list):
            for msg in messages:
                yield msg
        for value in payload.values():
            if value is messages:
                continue
            yield from _iter_messages(value)
        return
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, (dict, list)) or hasattr(item, "content"):
                yield from _iter_messages(item)
        return
    if hasattr(payload, "content") or hasattr(payload, "role") or hasattr(payload, "type"):
        yield payload


def stream_react_trace(agent, question: str, writer=None) -> bool:
    """流式打印 ReAct 轨迹。

    轨迹阶段：
    - THOUGHT: 可解释摘要（非原始思维链）
    - ACTION: 工具调用
    - OBSERVATION: 工具返回（可截断）
    - FINAL: 模型最终回答（token 流式输出）

    Returns:
        若成功输出任意内容则返回 True，否则返回 False。
    """
    stream_writer = writer or sys.stdout
    payload = {"messages": [("user", question)]}
    seen_actions: set[tuple[str, str]] = set()
    seen_thoughts: set[tuple[str, str]] = set()
    seen_observations: set[str] = set()
    final_started = False
    saw_token_stream = False
    fallback_final = ""
    wrote_anything = False

    def write_line(text: str) -> None:
        """输出一行并立即刷新。"""
        nonlocal wrote_anything
        stream_writer.write(text + "\n")
        if hasattr(stream_writer, "flush"):
            stream_writer.flush()
        wrote_anything = True

    def write_final_chunk(chunk: str) -> None:
        """按 token 片段输出 Final 阶段文本。"""
        nonlocal final_started, wrote_anything, saw_token_stream
        if not chunk:
            return
        if not final_started:
            stream_writer.write(f"{_stage_prefix('FINAL')} ")
            final_started = True
        stream_writer.write(chunk)
        if hasattr(stream_writer, "flush"):
            stream_writer.flush()
        wrote_anything = True
        saw_token_stream = True

    try:
        iterator = agent.stream(payload, stream_mode=["updates", "messages"])
    except Exception:
        return False

    for event in iterator:
        if isinstance(event, tuple) and len(event) == 2 and isinstance(event[0], str):
            mode, data = event
        else:
            mode, data = "updates", event

        if mode == "updates":
            for msg in _iter_messages(data):
                calls = _tool_calls(msg)
                for call in calls:
                    if isinstance(call, dict):
                        name = str(call.get("name") or "tool")
                        args = call.get("args", call.get("arguments", {}))
                    else:
                        name = str(getattr(call, "name", "tool"))
                        args = getattr(call, "args", getattr(call, "arguments", {}))
                    key = _action_key(name, args)
                    if key not in seen_actions:
                        if key not in seen_thoughts:
                            write_line(_thought_summary(args))
                            seen_thoughts.add(key)
                        line = _format_action(name, args)
                        write_line(line)
                        seen_actions.add(key)

                role = _msg_role(msg)
                if role in {"tool", "toolmessage"}:
                    observation = _content_to_text(_msg_value(msg, "content", ""))
                    short_obs = _truncate_non_final(observation)
                    line = f"{_stage_prefix('OBSERVATION')} {short_obs}"
                    if short_obs and line not in seen_observations:
                        write_line(line)
                        seen_observations.add(line)
                    continue

                if role in {"assistant", "ai", "aimessage"} and not calls:
                    text = _content_to_text(_msg_value(msg, "content", ""))
                    if text:
                        fallback_final = text

        elif mode == "messages":
            chunk = data
            if isinstance(data, tuple) and len(data) == 2:
                chunk, _ = data

            for call in _tool_call_chunks(chunk):
                if isinstance(call, dict):
                    name = str(call.get("name") or "tool")
                    args = call.get("args", call.get("arguments", ""))
                    if not args and call.get("args") is None:
                        args = call.get("args", "")
                else:
                    name = str(getattr(call, "name", "tool"))
                    args = getattr(call, "args", getattr(call, "arguments", ""))
                key = _action_key(name, args)
                if key not in seen_actions:
                    if key not in seen_thoughts:
                        write_line(_thought_summary(args))
                        seen_thoughts.add(key)
                    line = _format_action(name, args)
                    write_line(line)
                    seen_actions.add(key)

            role = _msg_role(chunk)
            if role in {"assistant", "ai", "aimessage"}:
                calls = _tool_calls(chunk)
                if not calls:
                    token = _content_to_text(_msg_value(chunk, "content", ""), strip=False)
                    write_final_chunk(token)
            elif role in {"tool", "toolmessage"}:
                observation = _content_to_text(_msg_value(chunk, "content", ""))
                short_obs = _truncate_non_final(observation)
                line = f"{_stage_prefix('OBSERVATION')} {short_obs}"
                if short_obs and line not in seen_observations:
                    write_line(line)
                    seen_observations.add(line)

    if not saw_token_stream and fallback_final:
        write_final_chunk(fallback_final)

    if final_started:
        stream_writer.write("\n")
        if hasattr(stream_writer, "flush"):
            stream_writer.flush()

    return wrote_anything


def cmd_ingest(args: argparse.Namespace, settings: Settings) -> int:
    """执行 `ingest` 子命令。"""
    count = build_faiss_index(
        docs_dir=args.docs_dir or settings.docs_dir,
        vector_store_dir=args.vector_store_dir or settings.vector_store_dir,
        embedding_model=args.embedding_model or settings.embedding_model,
        chunk_size=args.chunk_size or settings.chunk_size,
        chunk_overlap=args.chunk_overlap or settings.chunk_overlap,
    )
    print(json.dumps({"status": "ok", "chunks": count}, ensure_ascii=False))
    return 0


def cmd_chat(args: argparse.Namespace, settings: Settings) -> int:
    """执行 `chat` 子命令。"""
    try:
        from langchain_ollama import ChatOllama
    except ImportError as exc:
        raise RuntimeError("缺少 langchain-ollama，请先安装依赖") from exc

    retriever = FaissRetriever.from_disk(
        vector_store_dir=args.vector_store_dir or settings.vector_store_dir,
        embedding_model=args.embedding_model or settings.embedding_model,
        top_k=args.top_k or settings.rag_top_k,
    )
    llm = ChatOllama(
        model=args.model or settings.ollama_model,
        base_url=args.ollama_base_url or settings.ollama_base_url,
        temperature=0,
    )
    agent = build_react_agent(
        llm=llm,
        retriever=retriever,
        top_k=args.top_k or settings.rag_top_k,
    )

    if hasattr(agent, "stream") and stream_react_trace(agent, args.question):
        return 0

    result = agent.invoke({"messages": [("user", args.question)]})
    print(format_react_trace(result))
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI 主入口。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = Settings.from_env(base_dir=Path.cwd())

    if args.command == "ingest":
        return cmd_ingest(args, settings)
    if args.command == "chat":
        return cmd_chat(args, settings)
    parser.error(f"未知命令: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
