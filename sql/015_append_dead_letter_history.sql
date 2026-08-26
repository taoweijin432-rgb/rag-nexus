-- P6.1：把 Dead Letter 从“每个 Task 一条”升级为“每次 retry attempt 一条”。
-- 只修改本地 Index 控制表；不触碰四张 Source 表，也不写入业务数据。
-- 迁移可重复执行，部署前应在独立 Schema 双执行并核验。

SET @expert_index_dlq_table_exists = (
    SELECT COUNT(*)
    FROM information_schema.TABLES
    WHERE TABLE_SCHEMA = DATABASE()
      AND TABLE_NAME = 'expert_index_sync_dead_letter'
);

SET @expert_index_dlq_old_unique_exists = (
    SELECT COUNT(*)
    FROM information_schema.STATISTICS
    WHERE TABLE_SCHEMA = DATABASE()
      AND TABLE_NAME = 'expert_index_sync_dead_letter'
      AND INDEX_NAME = 'uk_expert_sync_dead_letter_task'
);

SET @expert_index_dlq_ddl = IF(
    @expert_index_dlq_table_exists = 1 AND @expert_index_dlq_old_unique_exists = 1,
    'ALTER TABLE expert_index_sync_dead_letter DROP INDEX uk_expert_sync_dead_letter_task',
    'SELECT 1'
);
PREPARE expert_index_dlq_stmt FROM @expert_index_dlq_ddl;
EXECUTE expert_index_dlq_stmt;
DEALLOCATE PREPARE expert_index_dlq_stmt;

SET @expert_index_dlq_attempt_unique_exists = (
    SELECT COUNT(*)
    FROM information_schema.STATISTICS
    WHERE TABLE_SCHEMA = DATABASE()
      AND TABLE_NAME = 'expert_index_sync_dead_letter'
      AND INDEX_NAME = 'uk_expert_sync_dead_letter_attempt'
);

SET @expert_index_dlq_ddl = IF(
    @expert_index_dlq_table_exists = 1 AND @expert_index_dlq_attempt_unique_exists = 0,
    'ALTER TABLE expert_index_sync_dead_letter ADD UNIQUE KEY uk_expert_sync_dead_letter_attempt (task_id, retry_count)',
    'SELECT 1'
);
PREPARE expert_index_dlq_stmt FROM @expert_index_dlq_ddl;
EXECUTE expert_index_dlq_stmt;
DEALLOCATE PREPARE expert_index_dlq_stmt;
