# XAgent VDB 后端扩展总设计与实施顺序（Architecture First）

## 1. 文档目的

- 基于现有三份梳理文档，给出“先做什么、后做什么”的完整执行顺序。
- 目标是把 Vector DB 做成可插拔/可选后端，同时先解决当前控制面与实现层错位问题。
- 本文只定义架构与实施路径，不包含具体代码实现。

## 2. 已知事实（来自代码与现有审计）

1. 当前运行主链路中，LanceDB 同时承担了控制面 metadata 与数据面向量索引。
2. Milvus/Chroma 目前主要是 provider 层实现，尚未接入完整 KB + Memory 主流程。
3. 需要迁移到 RDB 的 metadata 不止 `collection_metadata/main_pointers/prompt_templates/ingestion_runs`，还包括 `documents`，以及 `parses/chunks` 的控制字段（分阶段）。
4. 当前主要风险不是“缺 provider”，而是“抽象层与实现层错位 + 控制面与数据面耦合”。

结论：
- 必须先做“边界重构 + 控制面抽离”，再做多后端扩展。

## 3. 目标架构（终态）

- Control Plane (RDB)
- 元数据、流程状态、版本指针、模板、权限归属、集合/文档目录。

- Data Plane (VDB)
- 向量写入、索引管理、相似度检索、向量能力差异适配。

- Service 层
- 统一业务语义，组合 Control Plane + Data Plane，管理一致性与事务边界。

- Provider SPI 层
- `VectorIndexStore`：LanceDB/Milvus/Chroma 等实现。
- `MetadataStore`：RDB 实现（可先唯一实现）。

## 4. 总体原则（执行约束）

1. 先抽象边界，再改存储，再接新后端。
2. 控制面主权在 RDB，VDB 不再承载业务控制状态。
3. 先双写与回填，再切读，再停旧写，最后清理。
4. 每一步必须可灰度、可回滚、可观测。
5. 先 KB，后 Memory；先控制字段，后大字段。

## 5. 分阶段实施顺序（必须按序）

## Phase 0: 架构冻结（先做）

目标：统一定义，不写业务代码。

产物：
1. ADR-1: Control Plane / Data Plane 边界定义。
2. ADR-2: `MetadataStore` 与 `VectorIndexStore` 接口契约。
3. 字段归属清单（每个实体标注 RDB/VDB）。
4. 能力矩阵草案（LanceDB/Milvus/Chroma 的检索、过滤、索引差异）。

门禁（不通过不进入下一阶段）：
1. 边界中对 `documents/parses/chunks` 的归属已明确到字段级。
2. API 行为兼容策略已定（尤其 rename/delete/list/search）。
3. 迁移回滚策略书面化。

## Phase 1: 控制面防腐层（Anti-Corruption Layer）

目标：先切断业务层对 LanceDB 细节的直接依赖。

动作：
1. 在服务层引入 `MetadataStore` 接口与 façade。
2. 把控制面操作统一收口到接口（collection/doc/run/pointer/template）。
3. 现阶段可先由 LanceDB-backed adapter 承接，保证行为不变。

门禁：
1. `web/api/kb.py` 与 RAG management 不再直接拼 LanceDB 控制面语义。
2. 契约测试可覆盖控制面关键操作。

## Phase 2: RDB MetadataStore 落地 + 表结构上线

目标：把控制面真正落到 RDB，但先不切线上读路径。

动作：
1. 上线 `kb_*` 控制面表（按既有迁移文档）。
2. 实现 SQLAlchemy `MetadataStore`。
3. 接入 feature flag（读写策略可控）。

优先对象（本阶段必须完成）：
1. `collection_metadata`
2. `main_pointers`
3. `prompt_templates`
4. `ingestion_runs`
5. `documents`

门禁：
1. DDL 可回滚，读写单测齐全。
2. 与旧路径的行为对齐测试通过。

## Phase 3: 双写与历史回填

目标：建立一致性基础，验证新旧路径等价。

动作：
1. 控制面开启双写（RDB 主写 + VDB 兼容写）。
2. 回填历史数据（按 collection/doc 分批、可断点）。
3. 建立一致性巡检任务（计数+关键字段 hash + 抽样语义校验）。

门禁：
1. 一致性达到阈值（建议 >= 99.99%）。
2. 双写失败补偿链路可演练。

## Phase 4: 控制面读切换到 RDB

目标：业务读语义切换到 RDB，验证稳定性。

动作：
1. `KB_CONTROL_PLANE_READ_FROM_RDB=true` 灰度开启。
2. 逐步切换 collections/documents/status/pointers/templates 的读。
3. 观察窗口内持续巡检与告警。

门禁：
1. 功能与性能指标不劣化。
2. 回滚开关验证通过。

## Phase 5: 停止 VDB 控制面写入（完成解耦）

目标：完成“控制面与数据面分治”。

动作：
1. 关闭旧控制面表写入。
2. 保留短期只读兼容窗口。
3. 更新运维与故障手册。

门禁：
1. 线上无控制面回退依赖。
2. 迁移脚本、文档、监控全部切新路径。

## Phase 6: VectorIndexStore SPI 正式化与 LanceDB 对齐

目标：让“可插拔”从口号变成可运行能力。

动作：
1. 固化 `VectorIndexStore` 最小接口：upsert/delete/search/index/filter capabilities。
2. 先重构 LanceDB 实现完全对齐 SPI。
3. Service 层仅依赖 SPI，不感知具体后端。

门禁：
1. LanceDB 端到端回归全绿。
2. 不再有业务层直连 LanceDB 查询语法。

## Phase 7: Milvus/Chroma 渐进接入（后做）

目标：按能力矩阵逐个接入，避免一次性爆炸。

建议顺序：
1. Milvus（已有 provider 基础较好）
2. Chroma

动作：
1. 实现 SPI 适配。
2. 补 capability gap 的降级策略（如 sparse/hybrid/filter 差异）。
3. 新增后端选择配置与健康检查。

门禁：
1. 每个后端独立通过契约测试 + 集成测试。
2. 与 LanceDB baseline 的检索质量/性能差异在可接受阈值内。

## Phase 8: Phase-2 元数据对象迁移（parses/chunks）

目标：完成剩余 metadata 分层治理。

动作：
1. 先迁 `parses/chunks` 控制字段。
2. `parsed_content/chunk.text/chunk.metadata` 另行决策归宿（RDB 或对象存储）。

门禁：
1. 存储成本、查询性能、审计可追溯性三者达成平衡。

## 6. 为什么这个顺序是最优

1. 当前最大阻碍是耦合，不是 provider 数量。
2. 若先扩 Milvus/Chroma，再做边界重构，会把耦合复制到更多后端。
3. 先完成控制面分离后，多后端接入会变成“实现 SPI + 能力补齐”的线性工作。
4. 双写+回填+灰度切换是唯一可控的低风险路线。

## 7. 每阶段交付清单（DoD）

1. 架构文档：ADR、接口契约、字段归属表。
2. 数据文档：DDL、迁移计划、回填方案、回滚手册。
3. 测试文档：契约测试矩阵、端到端回归清单、验收阈值。
4. 运维文档：feature flag、监控指标、告警与应急预案。

## 8. 非目标（本轮不做）

1. 不在同一阶段同时重构 KB 与 Memory 的全部链路。
2. 不在 metadata 边界未冻结前推进大量 provider 特性开发。
3. 不在未建立一致性巡检前直接切主读路径。

## 9. 推荐启动动作（下一步）

1. 立即开展 Phase 0 评审会：冻结边界与契约。
2. 拆出两个并行设计包：
- A 包：MetadataStore（RDB）
- B 包：VectorIndexStore SPI（先 LanceDB 对齐）
3. 评审通过后再进入编码与迁移实施。

---

参考文档：
- `docs/backend-vector-db-control-plane-audit.md`
- `docs/vector_store/vdb-rdb-metadata-inventory.md`
- `docs/vector_store/vdb-rdb-metadata-migration-steps.md`
- `Issue #82` / `Issue #90`（方向共识）
