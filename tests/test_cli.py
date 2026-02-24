from io import StringIO

from react_agentic_rag.cli import build_parser, format_react_trace, stream_react_trace


def test_cli_has_ingest_and_chat_commands():
    parser = build_parser()
    choices = parser._subparsers._group_actions[0].choices

    assert "ingest" in choices
    assert "chat" in choices


def test_format_react_trace_outputs_action_observation_and_final():
    result = {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "name": "retrieve_context",
                        "args": {"query": "RAG流程"},
                    }
                ],
            },
            {
                "role": "tool",
                "name": "retrieve_context",
                "content": "source=docs/a.md chunk=a-1",
            },
            {
                "role": "assistant",
                "content": "这是最终答案",
            },
        ]
    }

    trace = format_react_trace(result)
    assert "[ACTION]" in trace
    assert "[THOUGHT]" in trace
    assert "retrieve_context" in trace
    assert "[OBSERVATION]" in trace
    assert "source=docs/a.md chunk=a-1" in trace
    assert "[FINAL]" in trace
    assert "这是最终答案" in trace


class _FakeStreamAgent:
    def stream(self, payload, stream_mode=None):
        assert payload["messages"][0][1] == "马海洋是谁"
        assert stream_mode == ["updates", "messages"]
        yield (
            "messages",
            (
                {
                    "role": "assistant",
                    "tool_call_chunks": [
                        {
                            "name": "retrieve_context",
                            "args": "{\"query\": \"\\u9a6c\\u6d77\\u6d0b\"}",
                        }
                    ],
                },
                {"langgraph_node": "agent"},
            ),
        )
        yield (
            "updates",
            {
                "agent": {
                    "messages": [
                        {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "name": "retrieve_context",
                                    "args": {"query": "马海洋"},
                                }
                            ],
                        }
                    ]
                }
            },
        )
        yield (
            "updates",
            {
                "tools": {
                    "messages": [
                        {
                            "role": "tool",
                            "content": "x" * 230,
                        }
                    ]
                }
            },
        )
        yield ("messages", ({"role": "assistant", "content": "马海洋"}, {"langgraph_node": "agent"}))
        yield ("messages", ({"role": "assistant", "content": "是Java后端工程师"}, {"langgraph_node": "agent"}))


def test_stream_react_trace_prints_realtime_sections():
    out = StringIO()
    ok = stream_react_trace(_FakeStreamAgent(), "马海洋是谁", out)

    text = out.getvalue()
    assert ok is True
    assert text.count("retrieve_context") == 1
    assert "\x1b[" in text
    assert text.count("[THOUGHT]") == 1
    assert "[ACTION]" in text
    assert "[OBSERVATION]" in text
    assert "[FINAL]" in text
    assert "...省略30字" in text
    assert "马海洋是Java后端工程师" in text
