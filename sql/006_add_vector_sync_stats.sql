-- 为现有派生索引库增加批次级向量同步累计统计。
--
-- 新建环境直接执行最新 001；已经存在 expert_index_batch 的环境执行本迁移。
-- 字段只保存数量，不保存 Chunk 正文、向量或专家个人信息。

ALTER TABLE expert_index_batch
    ADD COLUMN vector_sync_expert_count INT UNSIGNED NOT NULL DEFAULT 0
        COMMENT '已完成向量同步并返回可靠结果的专家数' AFTER rejected_count,
    ADD COLUMN vector_current_count INT UNSIGNED NOT NULL DEFAULT 0
        COMMENT '上游提交的当前 Chunk 总数' AFTER vector_sync_expert_count,
    ADD COLUMN vector_eligible_count INT UNSIGNED NOT NULL DEFAULT 0
        COMMENT '满足保留条件的 Chunk 总数' AFTER vector_current_count,
    ADD COLUMN vector_embedded_count INT UNSIGNED NOT NULL DEFAULT 0
        COMMENT '实际重新向量化的 Chunk 总数' AFTER vector_eligible_count,
    ADD COLUMN vector_metadata_updated_count INT UNSIGNED NOT NULL DEFAULT 0
        COMMENT '只更新 Metadata 的 Chunk 总数' AFTER vector_embedded_count,
    ADD COLUMN vector_unchanged_count INT UNSIGNED NOT NULL DEFAULT 0
        COMMENT '无需写入的未变化 Chunk 总数' AFTER vector_metadata_updated_count,
    ADD COLUMN vector_deleted_count INT UNSIGNED NOT NULL DEFAULT 0
        COMMENT '成功删除的旧向量总数' AFTER vector_unchanged_count,
    ADD COLUMN vector_delete_not_found_count INT UNSIGNED NOT NULL DEFAULT 0
        COMMENT '请求删除但下游未找到的向量总数' AFTER vector_deleted_count;
