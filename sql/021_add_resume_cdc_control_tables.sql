-- Resume CDC 控制面迁移：为既有 Resume Index Schema 增加 CDC 事件、业务水位，
-- 并允许标准同步任务使用 event_source='cdc'。
--
-- 强制边界：
-- 1. 只能在 RESUME_INDEX_DB_NAME 指向的独立 Resume Index Schema 执行；
-- 2. 不修改 Expert Index Schema，不连接或修改四张 Resume Source 表；
-- 3. 不写 Canal 水位、不创建 CDC 任务、不启动消费；上线前 RESUME_CDC_ENABLED 保持 0；
-- 4. 迁移可重复执行；所有业务表继续沿用复用 Repository 所需的 expert_index_* 表名。

-- 事件表只保存事件定位、resume_id 映射、处理状态和任务关联，不保存完整来源行。
CREATE TABLE IF NOT EXISTS expert_index_cdc_event (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    cdc_event_id VARCHAR(256) NULL COMMENT '由冻结身份策略生成的 CDC 事件唯一标识',
    local_batch_id BIGINT UNSIGNED NULL COMMENT 'Index 本地批次 ID，不是 Canal batch_id',
    task_id BIGINT UNSIGNED NULL COMMENT '逻辑关联 expert_index_sync_task.id',
    policy_version VARCHAR(32) NULL COMMENT 'CDC Task 身份使用的策略版本',
    destination VARCHAR(128) NOT NULL COMMENT '固定为 Resume 独立 destination company_resume',
    binlog_file VARCHAR(128) NOT NULL COMMENT '事件所在 MySQL Binlog 文件',
    binlog_position BIGINT UNSIGNED NOT NULL COMMENT '事件在 Binlog 文件中的偏移量',
    binlog_server_id BIGINT UNSIGNED NULL COMMENT 'Canal header server_id，仅用于协议审计',
    row_index INT UNSIGNED NOT NULL COMMENT '同一 Binlog 事件内从 0 开始的行序号',
    event_time DATETIME(6) NOT NULL COMMENT '来源 Binlog 事件发生时间',
    source_schema VARCHAR(128) NOT NULL COMMENT '发生变化的来源 Schema',
    source_table VARCHAR(128) NOT NULL COMMENT '发生变化的四张 Resume 来源表之一',
    operation_type VARCHAR(16) NOT NULL COMMENT 'insert/update/delete',
    source_record_id BIGINT UNSIGNED NOT NULL COMMENT '发生变化的来源表物理主键 id',
    expert_id BIGINT UNSIGNED NULL COMMENT '内部兼容字段，Resume 链路中固定承载 resume_id',
    status VARCHAR(16) NOT NULL DEFAULT 'pending'
        COMMENT 'pending/running/retry/success/dead_letter',
    attempt_count SMALLINT UNSIGNED NOT NULL DEFAULT 0 COMMENT '业务处理实际尝试次数',
    error_type VARCHAR(128) NULL COMMENT '安全异常类型，不保存来源行内容',
    error_message VARCHAR(512) NULL COMMENT '截断后的安全错误摘要，不保存 PII 或正文',
    processed_at DATETIME(6) NULL COMMENT '事件完成处理或进入 Dead Letter 的时间',
    created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6)
        ON UPDATE CURRENT_TIMESTAMP(6),
    PRIMARY KEY (id),
    UNIQUE KEY uk_expert_cdc_event_id (cdc_event_id),
    UNIQUE KEY uk_expert_cdc_event_position (
        destination, binlog_file, binlog_position, row_index
    ),
    KEY idx_expert_cdc_event_queue (status, id),
    KEY idx_expert_cdc_event_expert (expert_id, event_time),
    KEY idx_expert_cdc_event_task (task_id, status),
    CONSTRAINT chk_expert_cdc_event_table CHECK (
        source_table IN (
            'resume',
            'resume_work_experience',
            'resume_project_experience',
            'resume_education_experience'
        )
    ),
    CONSTRAINT chk_expert_cdc_event_operation CHECK (
        operation_type IN ('insert', 'update', 'delete')
    ),
    CONSTRAINT chk_expert_cdc_event_status CHECK (
        status IN ('pending', 'running', 'retry', 'success', 'dead_letter')
    )
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 业务 Checkpoint 只在 CDC 事件完成下游收敛后推进。本迁移只建表，不初始化水位。
CREATE TABLE IF NOT EXISTS expert_index_cdc_checkpoint (
    destination VARCHAR(128) NOT NULL COMMENT 'Resume 独立 Canal destination',
    binlog_server_id BIGINT UNSIGNED NOT NULL COMMENT '已应用位点的 MySQL server_id',
    binlog_file VARCHAR(128) NOT NULL COMMENT '已应用的 Binlog 文件',
    binlog_position BIGINT UNSIGNED NOT NULL COMMENT '已应用的 Binlog 文件内位置',
    last_local_batch_id BIGINT UNSIGNED NOT NULL COMMENT '最近提交的 Index 本地批次 ID',
    lock_version BIGINT UNSIGNED NOT NULL DEFAULT 0 COMMENT 'CAS 乐观锁版本',
    last_success_at DATETIME(6) NOT NULL COMMENT '最近一次成功提交的 Index MySQL 时间',
    created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6)
        ON UPDATE CURRENT_TIMESTAMP(6),
    PRIMARY KEY (destination),
    KEY idx_expert_cdc_checkpoint_position (
        binlog_server_id, binlog_file, binlog_position
    )
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 读取同步任务 event_source CHECK 的存在性与版本；只在旧约束缺少 cdc 时升级。
SET @resume_cdc_event_source_constraint_exists = (
    SELECT COUNT(*)
    FROM information_schema.CHECK_CONSTRAINTS cc
    INNER JOIN information_schema.TABLE_CONSTRAINTS tc
      ON tc.CONSTRAINT_SCHEMA = cc.CONSTRAINT_SCHEMA
     AND tc.CONSTRAINT_NAME = cc.CONSTRAINT_NAME
     AND tc.CONSTRAINT_TYPE = 'CHECK'
    WHERE cc.CONSTRAINT_SCHEMA = DATABASE()
      AND tc.TABLE_NAME = 'expert_index_sync_task'
      AND cc.CONSTRAINT_NAME = 'chk_expert_sync_task_event_source'
);

SET @resume_cdc_event_source_constraint_current = (
    SELECT CASE
        WHEN COUNT(*) = 1 AND MAX(cc.CHECK_CLAUSE) LIKE '%''cdc''%' THEN 1
        ELSE 0
    END
    FROM information_schema.CHECK_CONSTRAINTS cc
    INNER JOIN information_schema.TABLE_CONSTRAINTS tc
      ON tc.CONSTRAINT_SCHEMA = cc.CONSTRAINT_SCHEMA
     AND tc.CONSTRAINT_NAME = cc.CONSTRAINT_NAME
     AND tc.CONSTRAINT_TYPE = 'CHECK'
    WHERE cc.CONSTRAINT_SCHEMA = DATABASE()
      AND tc.TABLE_NAME = 'expert_index_sync_task'
      AND cc.CONSTRAINT_NAME = 'chk_expert_sync_task_event_source'
);

SET @resume_cdc_event_source_need_drop = (
    @resume_cdc_event_source_constraint_exists = 1
    AND @resume_cdc_event_source_constraint_current = 0
);
SET @resume_cdc_event_source_need_add = (
    @resume_cdc_event_source_constraint_current = 0
);

SET @resume_cdc_ddl = IF(
    @resume_cdc_event_source_need_drop = 1,
    'ALTER TABLE expert_index_sync_task DROP CHECK chk_expert_sync_task_event_source',
    'SELECT 1'
);
PREPARE resume_cdc_stmt FROM @resume_cdc_ddl;
EXECUTE resume_cdc_stmt;
DEALLOCATE PREPARE resume_cdc_stmt;

SET @resume_cdc_ddl = IF(
    @resume_cdc_event_source_need_add = 1,
    'ALTER TABLE expert_index_sync_task ADD CONSTRAINT chk_expert_sync_task_event_source CHECK (event_source IN (''poll'', ''cdc'', ''reconcile'', ''replay''))',
    'SELECT 1'
);
PREPARE resume_cdc_stmt FROM @resume_cdc_ddl;
EXECUTE resume_cdc_stmt;
DEALLOCATE PREPARE resume_cdc_stmt;
