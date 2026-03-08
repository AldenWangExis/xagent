# XAgent VDB -> RDB Metadata 迁移步骤（设计先行版）

## 1. 目标

- 将 KB 控制面 metadata 从 LanceDB 分离到 RDB。
- 保持向量检索数据仍在 VDB（LanceDB/Milvus/Chroma 可插拔）。
- 在不破坏现有功能的前提下，逐步切换读写路径。

## 2. 迁移对象分层

## 2.1 Phase-1 必迁对象

1. `collection_metadata`
2. `main_pointers`
3. `prompt_templates`
4. `ingestion_runs`
5. `documents`

## 2.2 Phase-2 分阶段对象

1. `parses`（先迁控制字段，后迁 `parsed_content`）
2. `chunks`（先迁控制字段，后迁 `text/metadata`）

## 2.3 保留在 VDB

1. `embeddings_{model_tag}`
2. memory `memories`

## 3. 目标数据模型（RDB）

建议新增（命名可调整）:

1. `kb_collections`
- `id`, `name`, `schema_version`, `embedding_model_id`, `embedding_dimension`
- `collection_locked`, `allow_mixed_parse_methods`, `skip_config_validation`
- `documents`, `processed_documents`, `parses`, `chunks`, `embeddings`
- `created_at`, `updated_at`, `last_accessed_at`, `extra_metadata`

2. `kb_documents`
- `id`, `collection_id`, `doc_id`, `source_path`, `file_type`, `content_hash`
- `title`, `language`, `uploaded_at`, `user_id`, `created_at`, `updated_at`

3. `kb_ingestion_runs`
- `id`, `collection_id`, `doc_id`, `status`, `message`, `parse_hash`, `user_id`
- `created_at`, `updated_at`

4. `kb_main_pointers`
- `id`, `collection_id`, `doc_id`, `step_type`, `model_tag`
- `semantic_id`, `technical_id`, `operator`, `created_at`, `updated_at`

5. `kb_prompt_templates`
- `id`, `collection_id`, `template_id`, `name`, `template`, `version`, `is_latest`
- `metadata`, `user_id`, `created_at`, `updated_at`

6. Phase-2:
- `kb_parses`, `kb_chunks`（至少先落控制字段）

7. 关系绑定治理:
- `agent_kb_bindings`（替代 `agents.knowledge_bases` 的名称引用）

## 4. 迁移阶段与执行步骤

## Phase 0: 架构冻结

1. 冻结边界: `Control Plane=RDB`, `Data Plane=VDB`。
2. 冻结接口: 定义控制面 repository 接口与向量后端 SPI。
3. 冻结一致性策略: 同步写主库 + 异步补偿（或双写校验）。

交付物:
- ADR/设计文档
- 字段映射表
- 回滚策略

## Phase 1: 建表与读写适配（不切流）

1. 新增 Alembic migration，创建上述 `kb_*` 表。
2. 新增 repository 实现（SQLAlchemy）。
3. 在现有服务层引入抽象，但读路径仍默认 VDB。

验收:
- 新表可创建、可回滚
- 单测通过

## Phase 2: 双写（VDB + RDB）

1. 对 `register_document`, `write_ingestion_status`, `main_pointer`, `prompt_template`, `collection_manager` 加双写。
2. 双写失败策略:
- RDB 写失败: 请求失败（控制面主权在 RDB）
- VDB 写失败: 标记补偿任务（避免在线阻塞）
3. 增加一致性巡检脚本（计数与抽样字段比对）。

验收:
- 双写期间一致性报告稳定
- 无明显性能退化

## Phase 3: 历史回填

1. 编写 backfill 脚本从 LanceDB 扫描历史数据写入 RDB。
2. 按 collection/doc 分批回填，支持断点续跑。
3. 回填后执行全量校验:
- 行数校验
- 关键字段哈希校验
- 抽样业务校验（list/get/stats/rename/delete）

验收:
- 一致性达标阈值通过（建议 >= 99.99%）

## Phase 4: 读路径切换

1. 开 feature flag:
- `KB_CONTROL_PLANE_READ_FROM_RDB=true`
2. 将这些读路径切到 RDB:
- collections/documents/stats/status/main pointers/prompt templates
3. 保留回退开关，观察一段稳定窗口。

验收:
- API 行为与旧路径等价
- 无 major 回归

## Phase 5: 写路径收敛与清理

1. 关闭 VDB 控制面写入（停止写 `collection_metadata/main_pointers/prompt_templates/ingestion_runs/documents`）。
2. 保留只读兼容窗口（可选），随后下线对应 LanceDB 控制面表。
3. 更新迁移脚本与运维文档，删除过时逻辑。

## Phase 6: Phase-2 对象迁移（可并行规划）

1. `parses/chunks` 控制字段迁移到 RDB。
2. 评估 `parsed_content`、`chunk.text/metadata` 的最终归宿:
- RDB（JSON/TEXT）
- 对象存储（推荐大对象）

## 5. 与可插拔向量后端的衔接

在完成 Phase 1~4 后，再推进向量后端 SPI:

1. `VectorBackend` 只负责:
- embeddings upsert/delete/query
- index capability
- filter capability
2. 不再承担控制面 metadata 语义。
3. LanceDB 先对齐新 SPI，再接 Milvus/Chroma 实现。

## 6. 测试与验收清单

1. 功能回归:
- ingest/search/list/delete/rename/retry/cancel
2. 一致性:
- VDB 与 RDB 双写比对
3. 稳定性:
- 并发写、重试、幂等
4. 安全:
- user_id 多租户隔离不退化
5. 迁移:
- backfill 可重入、可断点续跑、可回滚

## 7. 风险与回滚

主要风险:

1. 双写阶段的数据漂移
2. rename/delete 的跨存储原子性
3. 大字段回填导致窗口期过长

回滚策略:

1. 保留 `READ_FROM_RDB` 开关，出现异常立即回退读路径。
2. 保留 VDB 旧数据与写入逻辑直到稳定窗口结束。
3. 回填脚本按批次幂等设计，失败可重跑。

## 8. 建议实施顺序（简版）

1. 先做 `collection_metadata/main_pointers/prompt_templates/ingestion_runs/documents`
2. 再做 `parses/chunks` 控制字段
3. 最后清理历史 VDB 控制面表与名称绑定债务

