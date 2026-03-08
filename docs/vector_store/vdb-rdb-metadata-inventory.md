# XAgent VDB/RDB Metadata 全量梳理

## 1. 目标与范围

- 目标: 识别当前存放在 Vector DB (LanceDB) 的 metadata，判断哪些应迁移到 RDB。
- 范围: `src/xagent/core/tools/core/RAG_tools/*`、`src/xagent/web/*`、`src/xagent/core/memory/*`。
- 结论口径: 以代码行为为准，不以 issue 文本推断。

## 2. 当前 LanceDB 中的核心表

基于 `RAG_tools/LanceDB/schema_manager.py` 与调用方代码，当前主要表如下:

1. `collection_metadata`
2. `main_pointers`
3. `prompt_templates`
4. `ingestion_runs`
5. `documents`
6. `parses`
7. `chunks`
8. `embeddings_{model_tag}`

此外，memory 子系统独立使用 `memories`（`core/memory/lancedb.py` 默认 collection）。

## 3. RDB 现状

当前 SQLAlchemy 模型中没有 KB 专属控制面表。已有表主要是:

- 用户与模型: `users`, `models`, `user_models`, `user_default_models`
- 任务与追踪: `tasks`, `dag_executions`, `trace_events`
- Agent 配置: `agents`（其中 `knowledge_bases` 目前是 JSON 名称列表）
- 其他业务表: `mcp_servers`, `tool_configs`, `text2sql_databases` 等

结论: KB 控制面当前基本未落在 RDB。

## 4. 是否需要迁移的 metadata（判定）

## 4.1 必须迁移（控制面元数据）

1. `collection_metadata`
- 集合级配置与统计元信息，不是向量检索数据。

2. `main_pointers`
- 版本主指针，属于流程控制状态。

3. `prompt_templates`
- 提示词模板版本管理，属于业务配置。

4. `ingestion_runs`
- ingestion 状态流水，属于任务状态。

5. `documents`（新增必须迁移项）
- 文档注册元信息: `source_path/file_type/content_hash/uploaded_at/title/language/user_id`。
- 本质是文档目录与血缘信息，不是向量数据。

## 4.2 建议分阶段迁移（混合型）

1. `parses`
- 应迁字段（控制面）: `parse_hash/parser/params_json/created_at/user_id/collection/doc_id`
- 可后迁字段（大对象）: `parsed_content`

2. `chunks`
- 应迁字段（控制面）: `chunk_id/index/config_hash/chunk_hash/page_number/section/anchor/json_path/created_at/user_id/collection/doc_id/parse_hash`
- 可后迁字段（中间产物）: `text`, `metadata`

说明: `parses/chunks` 兼具中间数据与控制信息，建议先拆控制字段入 RDB，再按成本决定是否迁移大内容。

## 4.3 不建议迁移（数据面）

1. `embeddings_{model_tag}`
- 向量主体、检索索引、相似度搜索均依赖该层，属于数据面。

2. `memories`（memory 子系统）
- 业务向量记忆数据，不属于 KB 控制面 metadata。

## 4.4 非 VDB 但应同步治理

1. `agents.knowledge_bases`（RDB 中 JSON 名称列表）
- 当前按名称引用 KB，rename 场景容易漂移。
- 建议在 KB 控制面 RDB 化后改成关系绑定（如 `agent_kb_bindings`）。

## 5. 目标边界（建议）

- RDB (Control Plane): 集合、文档、版本指针、模板、状态、流程参数、权限归属。
- VDB (Data Plane): 向量、向量索引、检索必需最小字段（可含 `collection/doc_id/chunk_id` 外键键值）。

## 6. 最终结论

除了 `collection_metadata/main_pointers/prompt_templates/ingestion_runs` 之外，还应纳入迁移范围:

1. 必迁: `documents`
2. 分阶段迁移: `parses`（控制字段先迁）、`chunks`（控制字段先迁）
3. 同步治理: `agents.knowledge_bases` 的绑定方式

而 `embeddings_*` 与 memory `memories` 不应作为本轮 metadata 迁移目标。

