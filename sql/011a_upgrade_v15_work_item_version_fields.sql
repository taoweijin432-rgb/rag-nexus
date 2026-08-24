-- 011a：增量升级脚本 —— 把已应用 Task 1 版 011 的库补齐到当前 011 终态。
--
-- 背景：
--   d3ba3e6（T2-1）为 v15_vectorization_work_item 新增 chunk_type /
--   source_event_id / source_updated_at 三列、idx_v15_work_item_chunk 索引
--   与 chk_v15_work_item_chunk_type CHECK；但 011 使用
--   CREATE TABLE IF NOT EXISTS，已建表的数据库不会获得这些新对象。
--
-- 执行方式（禁止直接用 mysql client 裸跑，必须经安全门执行器）：
--   uv run python scripts/apply_v15_persistence_migration.py --upgrade
--
-- 适用范围：全新安装只需执行 011，不需要本脚本。
--
-- 幂等性：执行器按 information_schema 逐对象检查，已存在的对象自动跳过，
-- 重复运行安全。

ALTER TABLE v15_vectorization_work_item
    ADD COLUMN chunk_type VARCHAR(64) NULL COMMENT 'Contract chunk_type；disable 项可空' AFTER expert_id;

ALTER TABLE v15_vectorization_work_item
    ADD COLUMN source_event_id BIGINT UNSIGNED NULL COMMENT '上游 CDC 单调递增事件 id；新旧事件比较的主依据' AFTER source_field_id;

ALTER TABLE v15_vectorization_work_item
    ADD COLUMN source_updated_at DATETIME(6) NULL COMMENT '来源事件时间；仅作辅助排序，不做字典序比较' AFTER source_event_id;

ALTER TABLE v15_vectorization_work_item
    ADD KEY idx_v15_work_item_chunk (chunk_type, status);

ALTER TABLE v15_vectorization_work_item
    ADD CONSTRAINT chk_v15_work_item_chunk_type CHECK (
        chunk_type IS NULL OR chunk_type IN (
            'profile_summary', 'employment', 'project'
        )
    );
