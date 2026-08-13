-- 删除最终 expert_chunk 中产品确认不再需要的五个字段。
--
-- 适用范围：已经按 V1.3 扁平字段结构创建 expert_chunk 的现有数据库。
-- 新建数据库直接执行最新 001，不需要执行本脚本。
--
-- 注意：本次只调整最终 expert_chunk。Stage 表中的 source_item_key 与 data_version
-- 继续用于中间加工、来源分段和批次追踪，不在本迁移范围内。

ALTER TABLE expert_chunk
    DROP INDEX uk_expert_chunk_id,
    DROP INDEX idx_expert_chunk_source,
    DROP COLUMN chunk_id,
    DROP COLUMN token_count_estimate,
    DROP COLUMN source_item_key,
    DROP COLUMN dictionary_version,
    DROP COLUMN data_version,
    ADD KEY idx_expert_chunk_source (source_table, source_field_id);
