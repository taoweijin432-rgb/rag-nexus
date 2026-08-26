-- P6.3：补齐 DLQ 受控处置所需的操作者、请求身份和乐观锁版本。
-- 只修改本地 Index 控制表，不修改四张 Source 表，不写入业务数据。
-- 每个字段按 information_schema 幂等补建；部署前应在独立 Schema 双执行。

SET @expert_index_dlq_audit_table_exists = (
    SELECT COUNT(*)
    FROM information_schema.TABLES
    WHERE TABLE_SCHEMA = DATABASE()
      AND TABLE_NAME = 'expert_index_sync_dead_letter'
);

SET @expert_index_dlq_ddl = IF(
    @expert_index_dlq_audit_table_exists = 1
    AND NOT EXISTS (
        SELECT 1 FROM information_schema.COLUMNS
        WHERE TABLE_SCHEMA = DATABASE()
          AND TABLE_NAME = 'expert_index_sync_dead_letter'
          AND COLUMN_NAME = 'resolution_operator'
    ),
    'ALTER TABLE expert_index_sync_dead_letter ADD COLUMN resolution_operator VARCHAR(128) NULL COMMENT ''处置操作者安全标识'' AFTER resolution_note',
    'SELECT 1'
);
PREPARE expert_index_dlq_audit_stmt FROM @expert_index_dlq_ddl;
EXECUTE expert_index_dlq_audit_stmt;
DEALLOCATE PREPARE expert_index_dlq_audit_stmt;

SET @expert_index_dlq_ddl = IF(
    @expert_index_dlq_audit_table_exists = 1
    AND NOT EXISTS (
        SELECT 1 FROM information_schema.COLUMNS
        WHERE TABLE_SCHEMA = DATABASE()
          AND TABLE_NAME = 'expert_index_sync_dead_letter'
          AND COLUMN_NAME = 'resolution_request_id'
    ),
    'ALTER TABLE expert_index_sync_dead_letter ADD COLUMN resolution_request_id VARCHAR(128) NULL COMMENT ''处置请求幂等标识'' AFTER resolution_operator',
    'SELECT 1'
);
PREPARE expert_index_dlq_audit_stmt FROM @expert_index_dlq_ddl;
EXECUTE expert_index_dlq_audit_stmt;
DEALLOCATE PREPARE expert_index_dlq_audit_stmt;

SET @expert_index_dlq_ddl = IF(
    @expert_index_dlq_audit_table_exists = 1
    AND NOT EXISTS (
        SELECT 1 FROM information_schema.COLUMNS
        WHERE TABLE_SCHEMA = DATABASE()
          AND TABLE_NAME = 'expert_index_sync_dead_letter'
          AND COLUMN_NAME = 'resolution_version'
    ),
    'ALTER TABLE expert_index_sync_dead_letter ADD COLUMN resolution_version INT UNSIGNED NOT NULL DEFAULT 0 COMMENT ''处置状态乐观锁版本'' AFTER resolution_request_id',
    'SELECT 1'
);
PREPARE expert_index_dlq_audit_stmt FROM @expert_index_dlq_ddl;
EXECUTE expert_index_dlq_audit_stmt;
DEALLOCATE PREPARE expert_index_dlq_audit_stmt;
