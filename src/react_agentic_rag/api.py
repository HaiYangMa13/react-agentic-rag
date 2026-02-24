"""HTTP 服务入口。"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from react_agentic_rag.agent.react_graph import build_react_agent
from react_agentic_rag.config import Settings
from react_agentic_rag.rag.retriever import FaissRetriever


class ChatRequest(BaseModel):
    """聊天请求体。"""

    question: str
    top_k: int | None = None


class ChatResponse(BaseModel):
    """聊天响应体。"""

    answer: str


def create_app() -> FastAPI:
    """创建 FastAPI 应用并注册路由。"""
    app = FastAPI(title="react-agentic-rag", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        """健康检查接口。"""
        return {"status": "ok"}

    @app.post("/chat", response_model=ChatResponse)
    def chat(req: ChatRequest) -> ChatResponse:
        """执行单轮问答接口。"""
        settings = Settings.from_env(base_dir=Path.cwd())
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:
            raise HTTPException(status_code=500, detail="缺少 langchain-ollama 依赖") from exc

        try:
            retriever = FaissRetriever.from_disk(
                vector_store_dir=settings.vector_store_dir,
                embedding_model=settings.embedding_model,
                top_k=req.top_k or settings.rag_top_k,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"索引加载失败: {exc}") from exc

        llm = ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0,
        )
        agent = build_react_agent(
            llm=llm,
            retriever=retriever,
            top_k=req.top_k or settings.rag_top_k,
        )
        result = agent.invoke({"messages": [("user", req.question)]})
        answer = ""
        if isinstance(result, dict):
            messages = result.get("messages") or []
            if messages:
                last = messages[-1]
                if isinstance(last, dict):
                    answer = str(last.get("content", ""))
                else:
                    answer = str(last)
        if not answer:
            answer = str(result)
        return ChatResponse(answer=answer)

    return app


def run() -> None:
    """以脚本方式启动 API 服务。"""
    import uvicorn

    uvicorn.run(
        "react_agentic_rag.api:create_app",
        host="0.0.0.0",
        port=8000,
        factory=True,
        reload=False,
    )


app = create_app()
