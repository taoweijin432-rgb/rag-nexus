-- P7.4：补齐 CDC Event 幂等接收和标准 Task 关联字段。
--
-- 只修改本地 Index MySQL 的 expert_index_cdc_event，不修改四张 Source 表、
-- expert_chunk 或向量表。cdc_event_id 由已冻结（或测试显式注入）的身份策略提供，
-- 本迁移不定义 file/offset/row 的 canonical 计算方式。
--
-- 迁移前置：009_add_cdc_event_tracking.sql 已部署。
-- 幂等性：每个列/索引通过 information_schema 检查后再执行，重复执行安全。
-- 回滚：停止 CDC 接收器后保留字段；如确需 DROP，必须单独备份并审批。

SET @cdc_event_table_exists = (
    SELECT COUNT(*)
    FROM information_schema.TABLES
    WHERE TABLE_SCHEMA = DATABASE()
      AND TABLE_NAME = 'expert_index_cdc_event'
);

SET @ddl = IF(
    @cdc_event_table_exists = 1
    AND NOT EXISTS (
        SELECT 1 FROM information_schema.COLUMNS
        WHERE TABLE_SCHEMA = DATABASE()
          AND TABLE_NAME = 'expert_index_cdc_event'
          AND COLUMN_NAME = 'cdc_event_id'
    ),
    'ALTER TABLE expert_index_cdc_event ADD COLUMN cdc_event_id VARCHAR(256) NULL COMMENT ''由外部身份策略提供的 CDC 事件身份；未冻结时可为空'' AFTER id',
    'SELECT 1'
);
PREPARE stmt FROM @ddl; EXECUTE stmt; DEALLOCATE PREPARE stmt;

SET @ddl = IF(
    @cdc_event_table_exists = 1
    AND NOT EXISTS (
        SELECT 1 FROM information_schema.COLUMNS
        WHERE TABLE_SCHEMA = DATABASE()
          AND TABLE_NAME = 'expert_index_cdc_event'
          AND COLUMN_NAME = 'local_batch_id'
    ),
    'ALTER TABLE expert_index_cdc_event ADD COLUMN local_batch_id BIGINT UNSIGNED NULL COMMENT ''Index 本地批次 ID，不是 Canal batch_id'' AFTER cdc_event_id',
    'SELECT 1'
);
PREPARE stmt FROM @ddl; EXECUTE stmt; DEALLOCATE PREPARE stmt;

SET @ddl = IF(
    @cdc_event_table_exists = 1
    AND NOT EXISTS (
        SELECT 1 FROM information_schema.COLUMNS
        WHERE TABLE_SCHEMA = DATABASE()
          AND TABLE_NAME = 'expert_index_cdc_event'
          AND COLUMN_NAME = 'task_id'
    ),
    'ALTER TABLE expert_index_cdc_event ADD COLUMN task_id BIGINT UNSIGNED NULL COMMENT ''关联 expert_index_sync_task.id'' AFTER local_batch_id',
    'SELECT 1'
);
PREPARE stmt FROM @ddl; EXECUTE stmt; DEALLOCATE PREPARE stmt;

SET @ddl = IF(
    @cdc_event_table_exists = 1
    AND NOT EXISTS (
        SELECT 1 FROM information_schema.COLUMNS
        WHERE TABLE_SCHEMA = DATABASE()
          AND TABLE_NAME = 'expert_index_cdc_event'
          AND COLUMN_NAME = 'policy_version'
    ),
    'ALTER TABLE expert_index_cdc_event ADD COLUMN policy_version VARCHAR(32) NULL COMMENT ''CDC Task 身份使用的策略版本'' AFTER task_id',
    'SELECT 1'
);
PREPARE stmt FROM @ddl; EXECUTE stmt; DEALLOCATE PREPARE stmt;

SET @ddl = IF(
    @cdc_event_table_exists = 1
    AND NOT EXISTS (
        SELECT 1 FROM information_schema.COLUMNS
        WHERE TABLE_SCHEMA = DATABASE()
          AND TABLE_NAME = 'expert_index_cdc_event'
          AND COLUMN_NAME = 'binlog_server_id'
    ),
    'ALTER TABLE expert_index_cdc_event ADD COLUMN binlog_server_id BIGINT UNSIGNED NULL COMMENT ''Canal header server_id；仅作协议审计'' AFTER binlog_position',
    'SELECT 1'
);
PREPARE stmt FROM @ddl; EXECUTE stmt; DEALLOCATE PREPARE stmt;

SET @ddl = IF(
    @cdc_event_table_exists = 1
    AND NOT EXISTS (
        SELECT 1 FROM information_schema.STATISTICS
        WHERE TABLE_SCHEMA = DATABASE()
          AND TABLE_NAME = 'expert_index_cdc_event'
          AND INDEX_NAME = 'uk_expert_cdc_event_id'
    ),
    'ALTER TABLE expert_index_cdc_event ADD UNIQUE KEY uk_expert_cdc_event_id (cdc_event_id)',
    'SELECT 1'
);
PREPARE stmt FROM @ddl; EXECUTE stmt; DEALLOCATE PREPARE stmt;

SET @ddl = IF(
    @cdc_event_table_exists = 1
    AND NOT EXISTS (
        SELECT 1 FROM information_schema.STATISTICS
        WHERE TABLE_SCHEMA = DATABASE()
          AND TABLE_NAME = 'expert_index_cdc_event'
          AND INDEX_NAME = 'idx_expert_cdc_event_task'
    ),
    'ALTER TABLE expert_index_cdc_event ADD KEY idx_expert_cdc_event_task (task_id, status)',
    'SELECT 1'
);
PREPARE stmt FROM @ddl; EXECUTE stmt; DEALLOCATE PREPARE stmt;
