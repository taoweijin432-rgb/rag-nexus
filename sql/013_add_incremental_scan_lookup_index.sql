-- 为 P3 来源扫描恢复补充本地 Index MySQL 查询索引。
--
-- Scanner 在取得某张来源表的 Checkpoint lease 后，需要按 source_table 和未完成状态
-- 查找唯一可恢复的 Batch Source。现有索引均以 batch_id 开头，无法覆盖该查询。
-- 本迁移只增加二级索引，不修改四张来源表、不写控制表数据，也不推进 Checkpoint。
--
-- 降级时可以保留该兼容索引。若确需删除，必须先停止增量 Scanner、确认没有运行中
-- 查询，并单独执行 DROP INDEX；应用代码和控制表数据不需要回滚。
-- 若同名索引已经存在但定义错误，冲突分支会再次请求增加正确索引，由 MySQL 的
-- duplicate key name 错误中止迁移；迁移不会删除或静默覆盖未知结构。

SET @expert_index_ddl = (
    SELECT CASE
        WHEN COUNT(*) = 0 THEN
            'ALTER TABLE expert_index_batch_source ADD KEY idx_expert_batch_source_claim (source_table, status, id)'
        WHEN GROUP_CONCAT(COLUMN_NAME ORDER BY SEQ_IN_INDEX SEPARATOR ',') =
             'source_table,status,id'
             AND MIN(NON_UNIQUE) = 1 THEN
            'SELECT 1'
        ELSE
            'ALTER TABLE expert_index_batch_source ADD KEY idx_expert_batch_source_claim (source_table, status, id)'
    END
    FROM information_schema.STATISTICS
    WHERE TABLE_SCHEMA = DATABASE()
      AND TABLE_NAME = 'expert_index_batch_source'
      AND INDEX_NAME = 'idx_expert_batch_source_claim'
);
PREPARE expert_index_p3_stmt FROM @expert_index_ddl;
EXECUTE expert_index_p3_stmt;
DEALLOCATE PREPARE expert_index_p3_stmt;
