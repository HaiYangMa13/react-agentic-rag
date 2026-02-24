# ReAct + 本地 RAG 源码学习文档

本文按“从入口到执行链路”的方式解释当前项目代码，目标是让你能直接对照源码阅读。

## 1. 项目目标与运行形态

项目现在做了两件事：

1. 把文档离线向量化并存储到本地 FAISS。
2. 通过 ReAct Agent 在问答时调用检索工具，基于证据回答。

对应两个入口：

- CLI：`react-rag ingest` / `react-rag chat`
- API：`GET /health` / `POST /chat`

## 2. 目录职责

核心源码位于 `src/react_agentic_rag`：

- `config.py`：运行配置加载
- `rag/loader.py`：文档读取与文本抽取
- `rag/indexer.py`：切块与 FAISS 索引构建
- `rag/retriever.py`：FAISS 检索包装
- `agent/react_graph.py`：ReAct Agent 构建与 fallback
- `cli.py`：命令行入口与轨迹输出
- `api.py`：HTTP 接口入口
- `schemas.py`：结构化数据模型（当前偏预留）

## 3. 配置层（config.py）

`Settings` 用 dataclass 把所有关键参数集中管理：

- LLM：`OLLAMA_MODEL`、`OLLAMA_BASE_URL`
- RAG：`RAG_TOP_K`、`EMBEDDING_MODEL`
- 存储：`docs_dir`、`vector_store_dir`
- 切块：`CHUNK_SIZE`、`CHUNK_OVERLAP`

`from_env()` 的作用：

1. 读取环境变量；
2. 未设置时使用默认值；
3. 基于 `base_dir` 计算文档目录和索引目录。

这让 CLI / API / 测试共享同一套配置逻辑，避免散落硬编码。

## 4. 入库链路（ingest）

### 4.1 文档发现

`discover_documents(docs_dir)` 递归扫描目录，只保留 `{.txt,.md,.pdf}`。

### 4.2 文档读取

`load_document(path)`：

- `.txt/.md`：直接 UTF-8 读取；
- `.pdf`：通过 PyMuPDF(`fitz`) 抽取每页文本并拼接。

输出统一结构 `LoadedDocument(source, text)`。

### 4.3 切块

`chunk_text(text, chunk_size, chunk_overlap)` 是简化版滑窗切块：

- 先 `split()`；
- 每次取 `chunk_size`；
- 下一段回退 `chunk_overlap`。

优点是实现直观，便于学习；不足是对中文语义边界不够友好（后续可升级语义切分）。

### 4.4 向量化与落盘

`build_faiss_index(...)`：

1. 读取文档并切块；
2. 每个 chunk 打上 metadata：`source` + `chunk_id`；
3. 用 `HuggingFaceEmbeddings(model_name=...)` 生成向量；
4. `FAISS.from_documents(...)` 建库；
5. `save_local(vector_store_dir)` 持久化。

最终会在索引目录生成 FAISS 文件（如 `index.faiss` / `index.pkl`）。

## 5. 检索链路（retriever）

`FaissRetriever` 是对底层向量库的薄封装：

- `from_disk(...)`：加载本地索引；
- `search(query, top_k)`：执行 `similarity_search_with_score`，并映射为统一结构 `RetrievedChunk`。

`format_retrieval_output(chunks)` 把命中结果转成带引用的证据文本，供 LLM 直接消费。

## 6. Agent 编排（react_graph.py）

### 6.1 主路径：LangGraph ReAct

`build_react_agent(...)` 里会：

1. 定义工具函数 `retrieve_context(query)`；
2. 用 `@tool` 注册工具；
3. 调用 `create_react_agent(model, tools, prompt)` 构建 Agent。

系统提示词要求：

- 回答必须基于证据；
- 输出应包含来源；
- 证据不足时明确拒答。

### 6.2 兜底路径：_FallbackAgent

若 LangGraph 构建失败，返回 `_FallbackAgent`：

- `invoke()`：手动执行“检索 -> 拼提示 -> 调 LLM”；
- `stream()`：产出兼容事件，保证 CLI 仍可流式显示阶段信息。

兜底层的价值是：框架异常时仍可用，不会整条链路不可运行。

## 7. CLI 执行流程（cli.py）

### 7.1 命令解析

`build_parser()` 定义两个子命令：

- `ingest`：建索引
- `chat`：问答

### 7.2 chat 主流程

`cmd_chat()` 的执行顺序：

1. 初始化 `ChatOllama`；
2. 从磁盘加载 `FaissRetriever`；
3. `build_react_agent()` 获取 agent；
4. 优先走 `stream_react_trace(...)` 流式打印；
5. 若流式不可用，再回退 `invoke + format_react_trace`。

### 7.3 轨迹格式化（学习重点）

项目当前输出四段：

- `[THOUGHT]`：可解释摘要（不是原始 CoT）
- `[ACTION]`：工具调用
- `[OBSERVATION]`：工具结果（超长自动截断）
- `[FINAL]`：最终回答（流式 token）

其中：

- Action 参数会归一化，避免 `\uXXXX` 和中文重复打印；
- Observation 默认最多 200 字，尾部显示省略字数；
- 使用 ANSI 颜色做阶段区分。

## 8. API 执行流程（api.py）

`POST /chat` 的逻辑与 CLI 单轮问答一致：

1. 读取 `Settings`；
2. 加载 retriever；
3. 初始化 `ChatOllama`；
4. 构建 agent；
5. `invoke` 获取结果并提取最后一条回答文本。

当前 API 返回最小响应：`{"answer": "..."}`。

## 9. 数据对象（schemas.py）

`SourceCitation` 与 `AgentAnswer` 目前主要用于领域模型表达，尚未完全接入 API 响应结构。你后续做对外协议时可直接复用。

## 10. 测试覆盖说明

`tests/` 的测试重点是“接口契约”，不是模型效果评测：

- 配置默认值与环境覆盖
- 切块与文档发现规则
- ReAct 轨迹格式
- CLI 子命令存在
- 关键依赖声明

因此测试通过不代表“答案质量高”，只代表“工程流程可运行且行为符合当前约定”。

## 11. 你接下来最适合的学习顺序

建议按这个顺序打断点：

1. `cmd_ingest()` -> `build_faiss_index()`
2. `cmd_chat()` -> `build_react_agent()`
3. `stream_react_trace()` 观察事件如何被转换成阶段输出
4. `_FallbackAgent.stream()` 对比主路径事件结构

按这条线走，你能最快掌握“RAG 数据流 + Agent 编排 + 终端可观测性”。

## 12. 当前架构边界（已跑通，但尚未增强）

当前是教学和原型友好版本，还没做：

- 混合检索（BM25 + 向量）
- reranker 重排
- 多知识库隔离（`kb_id`）
- API 鉴权与限流
- 标准化评测框架

这部分建议在“可运行 + 可理解”之后分阶段推进。
