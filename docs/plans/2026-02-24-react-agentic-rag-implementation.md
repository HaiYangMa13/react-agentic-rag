# ReAct Agent + Agentic RAG Skeleton Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在本地离线环境中搭建可运行的 Python ReAct Agent 骨架，并预留 Agentic RAG 扩展接口。

**Architecture:** 使用 LangGraph 构建 ReAct 主循环，使用 Ollama 本地模型推理，提供 `retrieve_context` 工具连接 FAISS 向量检索。文档离线入库后由 CLI/API 统一调用 Agent 执行问答并返回引用来源。

**Tech Stack:** Python 3.11+, LangGraph, LangChain, langchain-ollama, FAISS, sentence-transformers, PyMuPDF, FastAPI, pytest。

### Task 1: 项目骨架与配置

**Files:**
- Create: `pyproject.toml`
- Create: `src/react_agentic_rag/config.py`
- Create: `src/react_agentic_rag/schemas.py`
- Test: `tests/test_config.py`

**Step 1: Write the failing test**
编写 `tests/test_config.py`，断言默认配置值与环境变量覆盖逻辑。

**Step 2: Run test to verify it fails**
Run: `pytest tests/test_config.py -v`
Expected: FAIL（模块未实现或导入失败）

**Step 3: Write minimal implementation**
实现 `Settings` 加载逻辑与基础数据结构。

**Step 4: Run test to verify it passes**
Run: `pytest tests/test_config.py -v`
Expected: PASS

### Task 2: 文档切块与向量索引

**Files:**
- Create: `src/react_agentic_rag/rag/loader.py`
- Create: `src/react_agentic_rag/rag/indexer.py`
- Test: `tests/test_indexer.py`

**Step 1: Write the failing test**
为纯文本切块、索引构建路径校验、空目录行为写测试。

**Step 2: Run test to verify it fails**
Run: `pytest tests/test_indexer.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**
实现文档读取、切块、FAISS 持久化构建。

**Step 4: Run test to verify it passes**
Run: `pytest tests/test_indexer.py -v`
Expected: PASS

### Task 3: ReAct Agent 与检索工具

**Files:**
- Create: `src/react_agentic_rag/rag/retriever.py`
- Create: `src/react_agentic_rag/agent/react_graph.py`
- Test: `tests/test_agent.py`

**Step 1: Write the failing test**
测试检索工具返回结构、Agent 构建函数返回可调用图对象。

**Step 2: Run test to verify it fails**
Run: `pytest tests/test_agent.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**
实现检索器、Tool、LangGraph ReAct 工厂函数。

**Step 4: Run test to verify it passes**
Run: `pytest tests/test_agent.py -v`
Expected: PASS

### Task 4: 运行入口（CLI/API）

**Files:**
- Create: `src/react_agentic_rag/cli.py`
- Create: `src/react_agentic_rag/api.py`
- Create: `scripts/ingest_docs.py`
- Create: `README.md`

**Step 1: Write the failing test**
为 CLI 基础命令参数解析写最小测试。

**Step 2: Run test to verify it fails**
Run: `pytest tests/test_cli.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**
实现 ingest/chat CLI 与 FastAPI `/chat`。

**Step 4: Run test to verify it passes**
Run: `pytest tests/test_cli.py -v`
Expected: PASS

### Task 5: 全量验证

**Files:**
- Modify: `README.md`

**Step 1: Run tests**
Run: `pytest -q`
Expected: 全部通过

**Step 2: Smoke test**
Run: `python -m react_agentic_rag.cli --help`
Expected: 正常打印帮助

**Step 3: Commit**
本轮按你的要求只落地代码，不自动提交。
