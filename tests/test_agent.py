from react_agentic_rag.agent.react_graph import build_react_agent
from react_agentic_rag.rag.retriever import RetrievedChunk, format_retrieval_output


class DummyRetriever:
    def search(self, query: str, top_k: int):
        return [
            RetrievedChunk(
                text=f"answer for {query}",
                source="docs/a.md",
                chunk_id="a-1",
                score=0.91,
            )
        ]


class DummyLLM:
    def invoke(self, messages):
        return {"messages": [{"role": "assistant", "content": "ok"}]}


def test_format_retrieval_output_includes_source():
    chunks = [
        RetrievedChunk(text="x", source="docs/x.md", chunk_id="x-1", score=0.8),
    ]
    output = format_retrieval_output(chunks)
    assert "docs/x.md" in output
    assert "x-1" in output


def test_build_react_agent_returns_runner():
    retriever = DummyRetriever()
    agent = build_react_agent(llm=DummyLLM(), retriever=retriever, top_k=2)
    assert hasattr(agent, "invoke")
    assert hasattr(agent, "stream")
