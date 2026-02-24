# react-agentic-rag

本地离线 ReAct Agent 骨架（Python），默认使用 Ollama + FAISS，已预留 Agentic RAG 扩展位。

## 1. 环境准备

```bash
cd /home/haiyang/code/react-agentic-rag
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

安装并启动 Ollama 后，拉取模型：

```bash
ollama pull qwen2.5:7b-instruct-q4_K_M
```

## 2. 放入文档

把文档放到 `data/docs/`（支持 `.txt` `.md` `.pdf`）。

## 3. 建索引

```bash
react-rag ingest
```

常用可选参数：

```bash
react-rag ingest --docs-dir data/docs --vector-store-dir data/vectorstore --chunk-size 400 --chunk-overlap 80
```

## 4. 聊天问答（ReAct）

```bash
react-rag chat --question "项目的核心流程是什么？"
```

## 5. 启动 API

```bash
react-rag-api
```

接口：

- `GET /health`
- `POST /chat`

示例：

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"question": "总结下文档要点"}'
```

## 6. Agentic RAG 扩展位

当前结构已拆分，后续可直接演进：

- `agent/react_graph.py`：加入 Planner / Verifier 子图
- `rag/retriever.py`：扩展混合召回（BM25 + 向量）
- `rag/indexer.py`：接入重排模型与元数据过滤

## 7. 测试

```bash
pytest -q
```
# react-agentic-rag
