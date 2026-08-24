-- 补齐 V2.2 增量编排所需的本地 Index MySQL 控制表。
--
-- 本迁移只创建或扩展 expert_index_* 状态表，不修改四张来源表、
-- expert_chunk、expert_index_cdc_event 或 011 的下游向量化表。
-- CREATE TABLE 用于补齐尚未部署的 007–010；后续 information_schema 检查
-- 用于幂等扩展已存在的旧表。迁移不创建任务，不推进水位。
--
-- 降级时应先停止增量 Worker，回退应用版本并保留本迁移的控制字段和失败证据。
-- 不建议自动 DROP 列或表；如确需物理回退，必须先备份 Index MySQL 并单独审批。

CREATE TABLE IF NOT EXISTS expert_index_source_checkpoint (
    source_table VARCHAR(128) NOT NULL COMMENT '物理来源表名，每张来源表独立维护一行',
    last_updated_at DATETIME(6) NOT NULL COMMENT '最后安全提交的来源更新时间游标',
    last_id BIGINT UNSIGNED NOT NULL COMMENT 'last_updated_at 相同时最后安全提交的来源主键',
    last_batch_id BIGINT UNSIGNED NULL COMMENT '最后成功推进 Checkpoint 的构建批次 ID',
    last_success_time DATETIME(6) NULL COMMENT '最近一次安全提交 Checkpoint 的系统时间',
    lock_version BIGINT UNSIGNED NOT NULL DEFAULT 0 COMMENT '防止旧批次覆盖新水位的乐观锁版本',
    lease_owner VARCHAR(128) NULL COMMENT '当前扫描租约持有者',
    lease_until DATETIME(6) NULL COMMENT '扫描租约到期时间，到期后可重新认领',
    created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
    PRIMARY KEY (source_table),
    KEY idx_expert_source_checkpoint_lease (lease_until, source_table),
    CONSTRAINT chk_expert_index_source_checkpoint_table CHECK (
        source_table IN (
            'experts',
            'experts_work_experience',
            'experts_project_experience',
            'experts_education_experience'
        )
    ),
    CONSTRAINT chk_expert_source_checkpoint_lease CHECK (
        (lease_owner IS NULL AND lease_until IS NULL)
        OR (lease_owner IS NOT NULL AND lease_until IS NOT NULL)
    )
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS expert_index_batch_source (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    batch_id BIGINT UNSIGNED NOT NULL COMMENT '逻辑关联 expert_index_batch.id',
    source_table VARCHAR(128) NOT NULL COMMENT '本行负责扫描的物理来源表',
    start_updated_at DATETIME(6) NOT NULL COMMENT '批次启动时的安全 Checkpoint 时间',
    start_id BIGINT UNSIGNED NOT NULL COMMENT '批次启动时的安全 Checkpoint 来源主键',
    cursor_updated_at DATETIME(6) NOT NULL COMMENT '当前已扫描到的临时时间游标',
    cursor_id BIGINT UNSIGNED NOT NULL COMMENT '当前已扫描到的临时来源主键游标',
    upper_bound_updated_at DATETIME(6) NOT NULL COMMENT '批次启动时固定的扫描上界时间',
    upper_bound_id BIGINT UNSIGNED NOT NULL COMMENT '批次启动时固定的来源主键上界',
    checkpoint_lock_version BIGINT UNSIGNED NOT NULL COMMENT '批次启动时读取的 Checkpoint 乐观锁版本',
    status VARCHAR(16) NOT NULL DEFAULT 'pending' COMMENT 'pending/running/success/failed',
    candidate_count INT UNSIGNED NOT NULL DEFAULT 0 COMMENT '该来源范围发现的候选记录数',
    success_count INT UNSIGNED NOT NULL DEFAULT 0 COMMENT '该来源范围已完成的任务数',
    failed_count INT UNSIGNED NOT NULL DEFAULT 0 COMMENT '该来源范围失败或进入 Dead Letter 的任务数',
    skipped_count INT UNSIGNED NOT NULL DEFAULT 0 COMMENT '该来源范围无需下游动作的任务数',
    started_at DATETIME(6) NULL COMMENT '开始扫描该来源表的时间',
    finished_at DATETIME(6) NULL COMMENT '该来源表批次到达终态的时间',
    created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
    PRIMARY KEY (id),
    UNIQUE KEY uk_expert_batch_source (batch_id, source_table),
    KEY idx_expert_batch_source_status (batch_id, status),
    CONSTRAINT chk_expert_batch_source_table CHECK (
        source_table IN (
            'experts',
            'experts_work_experience',
            'experts_project_experience',
            'experts_education_experience'
        )
    ),
    CONSTRAINT chk_expert_batch_source_status CHECK (
        status IN ('pending', 'running', 'success', 'failed')
    ),
    CONSTRAINT chk_expert_batch_source_start_cursor CHECK (
        start_updated_at < cursor_updated_at
        OR (start_updated_at = cursor_updated_at AND start_id <= cursor_id)
    ),
    CONSTRAINT chk_expert_batch_source_cursor_upper CHECK (
        cursor_updated_at < upper_bound_updated_at
        OR (cursor_updated_at = upper_bound_updated_at AND cursor_id <= upper_bound_id)
    )
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS expert_index_sync_task (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    dedupe_key CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL COMMENT '跨批次重复扫描幂等键',
    batch_id BIGINT UNSIGNED NOT NULL COMMENT '逻辑关联 expert_index_batch.id',
    event_source VARCHAR(16) NOT NULL DEFAULT 'poll' COMMENT 'poll/cdc/reconcile/replay',
    source_table VARCHAR(128) NOT NULL COMMENT '发生变化的四张专家来源表之一',
    source_record_id BIGINT UNSIGNED NOT NULL COMMENT '发生变化的来源表主键 id',
    source_updated_at DATETIME(6) NULL COMMENT '扫描候选的来源 updated_at，物理删除时可为空',
    expert_id BIGINT UNSIGNED NULL COMMENT '受影响的 experts.id',
    operation_type VARCHAR(16) NOT NULL COMMENT 'insert/update/delete',
    target_type VARCHAR(16) NOT NULL COMMENT 'expert/profile/employment/project/education',
    processing_stage VARCHAR(32) NOT NULL DEFAULT 'candidate' COMMENT '当前处理阶段',
    sync_status VARCHAR(16) NOT NULL DEFAULT 'pending' COMMENT 'pending/processing/success/failed/retrying/dead',
    retry_count SMALLINT UNSIGNED NOT NULL DEFAULT 0 COMMENT '自动重试次数，不包含首次处理',
    attempt_token VARCHAR(64) NULL COMMENT '当前认领尝试的唯一令牌',
    lease_until DATETIME(6) NULL COMMENT '当前任务租约到期时间',
    next_retry_at DATETIME(6) NULL COMMENT '下一次允许认领的时间',
    policy_version VARCHAR(32) NULL COMMENT '产生幂等键和动作计划时的策略版本',
    error_stage VARCHAR(32) NULL COMMENT '最近失败所在阶段',
    error_code VARCHAR(64) NULL COMMENT '稳定机器错误码',
    error_type VARCHAR(128) NULL COMMENT '安全异常类型',
    error_message VARCHAR(512) NULL COMMENT '截断后的安全错误摘要',
    result_code VARCHAR(64) NULL COMMENT '机器处理结果码',
    started_at DATETIME(6) NULL COMMENT '最近一次开始处理的时间',
    completed_at DATETIME(6) NULL COMMENT '进入最终终态的时间',
    created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
    PRIMARY KEY (id),
    UNIQUE KEY uk_expert_sync_task_dedupe (dedupe_key),
    UNIQUE KEY uk_expert_sync_task_source (batch_id, source_table, source_record_id),
    KEY idx_expert_sync_task_queue (sync_status, id),
    KEY idx_expert_sync_task_retry_queue (sync_status, next_retry_at, id),
    KEY idx_expert_sync_task_batch (batch_id, sync_status, id),
    KEY idx_expert_sync_task_expert (expert_id, id),
    KEY idx_expert_sync_task_history (expert_id, created_at),
    KEY idx_expert_sync_task_source_locator (source_table, source_record_id, source_updated_at),
    CONSTRAINT chk_expert_sync_task_table CHECK (
        source_table IN (
            'experts',
            'experts_work_experience',
            'experts_project_experience',
            'experts_education_experience'
        )
    ),
    CONSTRAINT chk_expert_sync_task_operation CHECK (
        operation_type IN ('insert', 'update', 'delete')
    ),
    CONSTRAINT chk_expert_sync_task_event_source CHECK (
        event_source IN ('poll', 'cdc', 'reconcile', 'replay')
    ),
    CONSTRAINT chk_expert_sync_task_target CHECK (
        target_type IN ('expert', 'profile', 'employment', 'project', 'education')
    ),
    CONSTRAINT chk_expert_sync_task_stage CHECK (
        processing_stage IN (
            'candidate',
            'source_read',
            'normalized',
            'chunk_built',
            'validated',
            'contract_ready',
            'handed_off',
            'completed'
        )
    ),
    CONSTRAINT chk_expert_sync_task_status CHECK (
        sync_status IN ('pending', 'processing', 'success', 'failed', 'retrying', 'dead')
    ),
    CONSTRAINT chk_expert_sync_task_attempt_lease CHECK (
        (attempt_token IS NULL AND lease_until IS NULL)
        OR (attempt_token IS NOT NULL AND lease_until IS NOT NULL)
    ),
    CONSTRAINT chk_expert_sync_task_error_stage CHECK (
        error_stage IS NULL OR error_stage IN (
            'source',
            'parse',
            'pii',
            'chunk_builder',
            'generation',
            'validation',
            'embedding',
            'milvus',
            'checkpoint'
        )
    )
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS expert_index_sync_artifact (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    task_id BIGINT UNSIGNED NOT NULL COMMENT '逻辑关联 expert_index_sync_task.id',
    artifact_type VARCHAR(32) NOT NULL COMMENT 'contract_payload/contract_reference',
    artifact_version VARCHAR(32) NOT NULL COMMENT '恢复 Payload 的 Contract 版本',
    payload_json JSON NULL COMMENT '通过白名单和 PII 校验的受控 Contract Payload',
    payload_reference VARCHAR(512) NULL COMMENT '可恢复受控 Contract Payload 的引用',
    payload_hash CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL COMMENT 'Payload 完整性 SHA-256',
    expires_at DATETIME(6) NULL COMMENT '恢复证据的过期时间',
    created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
    PRIMARY KEY (id),
    UNIQUE KEY uk_expert_sync_artifact_version (task_id, artifact_type, artifact_version),
    KEY idx_expert_sync_artifact_expiry (expires_at, id),
    CONSTRAINT chk_expert_sync_artifact_type CHECK (
        artifact_type IN ('contract_payload', 'contract_reference')
    ),
    CONSTRAINT chk_expert_sync_artifact_payload CHECK (
        (payload_json IS NOT NULL AND payload_reference IS NULL)
        OR (payload_json IS NULL AND payload_reference IS NOT NULL)
    )
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS expert_index_sync_dead_letter (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    task_id BIGINT UNSIGNED NOT NULL COMMENT '逻辑关联 expert_index_sync_task.id',
    batch_id BIGINT UNSIGNED NOT NULL COMMENT '冗余所属批次，便于审计和补偿筛选',
    source_table VARCHAR(128) NOT NULL COMMENT '发生失败的物理来源表',
    source_record_id BIGINT UNSIGNED NOT NULL COMMENT '发生失败的来源表主键 id',
    expert_id BIGINT UNSIGNED NULL COMMENT '受影响的 experts.id',
    failure_stage VARCHAR(32) NOT NULL COMMENT '进入 Dead Letter 时的处理阶段',
    reason_code VARCHAR(64) NOT NULL COMMENT '稳定机器失败码',
    reason_message VARCHAR(512) NULL COMMENT '截断后的安全失败摘要',
    retry_count SMALLINT UNSIGNED NOT NULL COMMENT '进入 Dead Letter 时的自动重试次数',
    artifact_id BIGINT UNSIGNED NULL COMMENT '逻辑关联可用的受控恢复证据',
    resolution_status VARCHAR(16) NOT NULL DEFAULT 'pending' COMMENT 'pending/requeued/resolved/ignored',
    resolution_note VARCHAR(512) NULL COMMENT '截断后的安全处置说明',
    resolved_at DATETIME(6) NULL COMMENT '人工或专项补偿完成时间',
    created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
    PRIMARY KEY (id),
    UNIQUE KEY uk_expert_sync_dead_letter_task (task_id),
    KEY idx_expert_sync_dead_letter_resolution (resolution_status, created_at),
    KEY idx_expert_sync_dead_letter_batch (batch_id, id),
    KEY idx_expert_sync_dead_letter_expert (expert_id, id),
    CONSTRAINT chk_expert_sync_dead_letter_table CHECK (
        source_table IN (
            'experts',
            'experts_work_experience',
            'experts_project_experience',
            'experts_education_experience'
        )
    ),
    CONSTRAINT chk_expert_sync_dead_letter_resolution CHECK (
        resolution_status IN ('pending', 'requeued', 'resolved', 'ignored')
    ),
    CONSTRAINT chk_expert_sync_dead_letter_resolved_at CHECK (
        (resolution_status = 'pending' AND resolved_at IS NULL)
        OR (resolution_status IN ('requeued', 'resolved', 'ignored') AND resolved_at IS NOT NULL)
    )
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 已有 007 表只在字段缺失时补充租约能力。
SET @expert_index_ddl = (
    SELECT IF(
        COUNT(*) = 0,
        'ALTER TABLE expert_index_source_checkpoint ADD COLUMN lease_owner VARCHAR(128) NULL COMMENT ''当前扫描租约持有者'' AFTER lock_version',
        'SELECT 1'
    )
    FROM information_schema.COLUMNS
    WHERE TABLE_SCHEMA = DATABASE()
      AND TABLE_NAME = 'expert_index_source_checkpoint'
      AND COLUMN_NAME = 'lease_owner'
);
PREPARE expert_index_p1_stmt FROM @expert_index_ddl;
EXECUTE expert_index_p1_stmt;
DEALLOCATE PREPARE expert_index_p1_stmt;

SET @expert_index_ddl = (
    SELECT IF(
        COUNT(*) = 0,
        'ALTER TABLE expert_index_source_checkpoint ADD COLUMN lease_until DATETIME(6) NULL COMMENT ''扫描租约到期时间，到期后可重新认领'' AFTER lease_owner',
        'SELECT 1'
    )
    FROM information_schema.COLUMNS
    WHERE TABLE_SCHEMA = DATABASE()
      AND TABLE_NAME = 'expert_index_source_checkpoint'
      AND COLUMN_NAME = 'lease_until'
);
PREPARE expert_index_p1_stmt FROM @expert_index_ddl;
EXECUTE expert_index_p1_stmt;
DEALLOCATE PREPARE expert_index_p1_stmt;

SET @expert_index_ddl = (
    SELECT IF(
        COUNT(*) = 0,
        'ALTER TABLE expert_index_source_checkpoint ADD KEY idx_expert_source_checkpoint_lease (lease_until, source_table)',
        'SELECT 1'
    )
    FROM information_schema.STATISTICS
    WHERE TABLE_SCHEMA = DATABASE()
      AND TABLE_NAME = 'expert_index_source_checkpoint'
      AND INDEX_NAME = 'idx_expert_source_checkpoint_lease'
);
PREPARE expert_index_p1_stmt FROM @expert_index_ddl;
EXECUTE expert_index_p1_stmt;
DEALLOCATE PREPARE expert_index_p1_stmt;

SET @expert_index_ddl = (
    SELECT IF(
        COUNT(*) = 0,
        'ALTER TABLE expert_index_source_checkpoint ADD CONSTRAINT chk_expert_source_checkpoint_lease CHECK ((lease_owner IS NULL AND lease_until IS NULL) OR (lease_owner IS NOT NULL AND lease_until IS NOT NULL))',
        'SELECT 1'
    )
    FROM information_schema.TABLE_CONSTRAINTS
    WHERE CONSTRAINT_SCHEMA = DATABASE()
      AND TABLE_NAME = 'expert_index_source_checkpoint'
      AND CONSTRAINT_NAME = 'chk_expert_source_checkpoint_lease'
);
PREPARE expert_index_p1_stmt FROM @expert_index_ddl;
EXECUTE expert_index_p1_stmt;
DEALLOCATE PREPARE expert_index_p1_stmt;

-- 已有 008 表只补充来源表级处理计数。
SET @expert_index_ddl = (
    SELECT IF(
        COUNT(*) = 0,
        'ALTER TABLE expert_index_batch_source ADD COLUMN candidate_count INT UNSIGNED NOT NULL DEFAULT 0 COMMENT ''该来源范围发现的候选记录数'' AFTER status',
        'SELECT 1'
    )
    FROM information_schema.COLUMNS
    WHERE TABLE_SCHEMA = DATABASE()
      AND TABLE_NAME = 'expert_index_batch_source'
      AND COLUMN_NAME = 'candidate_count'
);
PREPARE expert_index_p1_stmt FROM @expert_index_ddl;
EXECUTE expert_index_p1_stmt;
DEALLOCATE PREPARE expert_index_p1_stmt;

SET @expert_index_ddl = (
    SELECT IF(
        COUNT(*) = 0,
        'ALTER TABLE expert_index_batch_source ADD COLUMN success_count INT UNSIGNED NOT NULL DEFAULT 0 COMMENT ''该来源范围已完成的任务数'' AFTER candidate_count',
        'SELECT 1'
    )
    FROM information_schema.COLUMNS
    WHERE TABLE_SCHEMA = DATABASE()
      AND TABLE_NAME = 'expert_index_batch_source'
      AND COLUMN_NAME = 'success_count'
);
PREPARE expert_index_p1_stmt FROM @expert_index_ddl;
EXECUTE expert_index_p1_stmt;
DEALLOCATE PREPARE expert_index_p1_stmt;

SET @expert_index_ddl = (
    SELECT IF(
        COUNT(*) = 0,
        'ALTER TABLE expert_index_batch_source ADD COLUMN failed_count INT UNSIGNED NOT NULL DEFAULT 0 COMMENT ''该来源范围失败或进入 Dead Letter 的任务数'' AFTER success_count',
        'SELECT 1'
    )
    FROM information_schema.COLUMNS
    WHERE TABLE_SCHEMA = DATABASE()
      AND TABLE_NAME = 'expert_index_batch_source'
      AND COLUMN_NAME = 'failed_count'
);
PREPARE expert_index_p1_stmt FROM @expert_index_ddl;
EXECUTE expert_index_p1_stmt;
DEALLOCATE PREPARE expert_index_p1_stmt;

SET @expert_index_ddl = (
    SELECT IF(
        COUNT(*) = 0,
        'ALTER TABLE expert_index_batch_source ADD COLUMN skipped_count INT UNSIGNED NOT NULL DEFAULT 0 COMMENT ''该来源范围无需下游动作的任务数'' AFTER failed_count',
        'SELECT 1'
    )
    FROM information_schema.COLUMNS
    WHERE TABLE_SCHEMA = DATABASE()
      AND TABLE_NAME = 'expert_index_batch_source'
      AND COLUMN_NAME = 'skipped_count'
);
PREPARE expert_index_p1_stmt FROM @expert_index_ddl;
EXECUTE expert_index_p1_stmt;
DEALLOCATE PREPARE expert_index_p1_stmt;

-- 已有 010 表保留原字段和批次内唯一键，只补充跨扫描幂等、租约和重试字段。
SET @expert_index_ddl = (
    SELECT IF(
        COUNT(*) = 0,
        'ALTER TABLE expert_index_sync_task ADD COLUMN dedupe_key CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NULL COMMENT ''跨批次重复扫描幂等键'' AFTER id',
        'SELECT 1'
    )
    FROM information_schema.COLUMNS
    WHERE TABLE_SCHEMA = DATABASE()
      AND TABLE_NAME = 'expert_index_sync_task'
      AND COLUMN_NAME = 'dedupe_key'
);
PREPARE expert_index_p1_stmt FROM @expert_index_ddl;
EXECUTE expert_index_p1_stmt;
DEALLOCATE PREPARE expert_index_p1_stmt;

SET @expert_index_ddl = (
    SELECT IF(COUNT(*) = 0,
        'ALTER TABLE expert_index_sync_task ADD COLUMN event_source VARCHAR(16) NOT NULL DEFAULT ''poll'' COMMENT ''poll/cdc/reconcile/replay'' AFTER batch_id',
        'SELECT 1')
    FROM information_schema.COLUMNS
    WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'expert_index_sync_task' AND COLUMN_NAME = 'event_source'
);
PREPARE expert_index_p1_stmt FROM @expert_index_ddl;
EXECUTE expert_index_p1_stmt;
DEALLOCATE PREPARE expert_index_p1_stmt;

SET @expert_index_ddl = (
    SELECT IF(COUNT(*) = 0,
        'ALTER TABLE expert_index_sync_task ADD COLUMN source_updated_at DATETIME(6) NULL COMMENT ''扫描候选的来源 updated_at，物理删除时可为空'' AFTER source_record_id',
        'SELECT 1')
    FROM information_schema.COLUMNS
    WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'expert_index_sync_task' AND COLUMN_NAME = 'source_updated_at'
);
PREPARE expert_index_p1_stmt FROM @expert_index_ddl;
EXECUTE expert_index_p1_stmt;
DEALLOCATE PREPARE expert_index_p1_stmt;

SET @expert_index_ddl = (
    SELECT IF(COUNT(*) = 0,
        'ALTER TABLE expert_index_sync_task ADD COLUMN target_type VARCHAR(16) NULL COMMENT ''expert/profile/employment/project/education'' AFTER operation_type',
        'SELECT 1')
    FROM information_schema.COLUMNS
    WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'expert_index_sync_task' AND COLUMN_NAME = 'target_type'
);
PREPARE expert_index_p1_stmt FROM @expert_index_ddl;
EXECUTE expert_index_p1_stmt;
DEALLOCATE PREPARE expert_index_p1_stmt;

SET @expert_index_ddl = (
    SELECT IF(COUNT(*) = 0,
        'ALTER TABLE expert_index_sync_task ADD COLUMN processing_stage VARCHAR(32) NOT NULL DEFAULT ''candidate'' COMMENT ''当前处理阶段'' AFTER target_type',
        'SELECT 1')
    FROM information_schema.COLUMNS
    WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'expert_index_sync_task' AND COLUMN_NAME = 'processing_stage'
);
PREPARE expert_index_p1_stmt FROM @expert_index_ddl;
EXECUTE expert_index_p1_stmt;
DEALLOCATE PREPARE expert_index_p1_stmt;

SET @expert_index_ddl = (
    SELECT IF(COUNT(*) = 0,
        'ALTER TABLE expert_index_sync_task ADD COLUMN attempt_token VARCHAR(64) NULL COMMENT ''当前认领尝试的唯一令牌'' AFTER retry_count',
        'SELECT 1')
    FROM information_schema.COLUMNS
    WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'expert_index_sync_task' AND COLUMN_NAME = 'attempt_token'
);
PREPARE expert_index_p1_stmt FROM @expert_index_ddl;
EXECUTE expert_index_p1_stmt;
DEALLOCATE PREPARE expert_index_p1_stmt;

SET @expert_index_ddl = (
    SELECT IF(COUNT(*) = 0,
        'ALTER TABLE expert_index_sync_task ADD COLUMN lease_until DATETIME(6) NULL COMMENT ''当前任务租约到期时间'' AFTER attempt_token',
        'SELECT 1')
    FROM information_schema.COLUMNS
    WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'expert_index_sync_task' AND COLUMN_NAME = 'lease_until'
);
PREPARE expert_index_p1_stmt FROM @expert_index_ddl;
EXECUTE expert_index_p1_stmt;
DEALLOCATE PREPARE expert_index_p1_stmt;

SET @expert_index_ddl = (
    SELECT IF(COUNT(*) = 0,
        'ALTER TABLE expert_index_sync_task ADD COLUMN next_retry_at DATETIME(6) NULL COMMENT ''下一次允许认领的时间'' AFTER lease_until',
        'SELECT 1')
    FROM information_schema.COLUMNS
    WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'expert_index_sync_task' AND COLUMN_NAME = 'next_retry_at'
);
PREPARE expert_index_p1_stmt FROM @expert_index_ddl;
EXECUTE expert_index_p1_stmt;
DEALLOCATE PREPARE expert_index_p1_stmt;

SET @expert_index_ddl = (
    SELECT IF(COUNT(*) = 0,
        'ALTER TABLE expert_index_sync_task ADD COLUMN policy_version VARCHAR(32) NULL COMMENT ''产生幂等键和动作计划时的策略版本'' AFTER next_retry_at',
        'SELECT 1')
    FROM information_schema.COLUMNS
    WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'expert_index_sync_task' AND COLUMN_NAME = 'policy_version'
);
PREPARE expert_index_p1_stmt FROM @expert_index_ddl;
EXECUTE expert_index_p1_stmt;
DEALLOCATE PREPARE expert_index_p1_stmt;

SET @expert_index_ddl = (
    SELECT IF(COUNT(*) = 0,
        'ALTER TABLE expert_index_sync_task ADD COLUMN error_code VARCHAR(64) NULL COMMENT ''稳定机器错误码'' AFTER error_stage',
        'SELECT 1')
    FROM information_schema.COLUMNS
    WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'expert_index_sync_task' AND COLUMN_NAME = 'error_code'
);
PREPARE expert_index_p1_stmt FROM @expert_index_ddl;
EXECUTE expert_index_p1_stmt;
DEALLOCATE PREPARE expert_index_p1_stmt;

SET @expert_index_ddl = (
    SELECT IF(COUNT(*) = 0,
        'ALTER TABLE expert_index_sync_task ADD COLUMN updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6) AFTER created_at',
        'SELECT 1')
    FROM information_schema.COLUMNS
    WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'expert_index_sync_task' AND COLUMN_NAME = 'updated_at'
);
PREPARE expert_index_p1_stmt FROM @expert_index_ddl;
EXECUTE expert_index_p1_stmt;
DEALLOCATE PREPARE expert_index_p1_stmt;

-- 旧 010 任务缺少新幂等公式所需的来源时间和策略版本。
-- 为它们生成只用于迁移兼容的稳定唯一值，新任务必须由业务代码按冻结公式生成。
UPDATE expert_index_sync_task
SET dedupe_key = SHA2(
        CONCAT_WS(
            '|',
            'legacy',
            CAST(id AS CHAR),
            CAST(batch_id AS CHAR),
            source_table,
            CAST(source_record_id AS CHAR),
            operation_type
        ),
        256
    )
WHERE dedupe_key IS NULL;

UPDATE expert_index_sync_task
SET target_type = CASE
        WHEN source_table = 'experts' AND operation_type = 'delete' THEN 'expert'
        WHEN source_table = 'experts' THEN 'profile'
        WHEN source_table = 'experts_work_experience' THEN 'employment'
        WHEN source_table = 'experts_project_experience' THEN 'project'
        WHEN source_table = 'experts_education_experience' THEN 'education'
    END
WHERE target_type IS NULL;

SET @expert_index_ddl = (
    SELECT IF(
        COUNT(*) = 1,
        'ALTER TABLE expert_index_sync_task MODIFY COLUMN dedupe_key CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL COMMENT ''跨批次重复扫描幂等键''',
        'SELECT 1'
    )
    FROM information_schema.COLUMNS
    WHERE TABLE_SCHEMA = DATABASE()
      AND TABLE_NAME = 'expert_index_sync_task'
      AND COLUMN_NAME = 'dedupe_key'
      AND IS_NULLABLE = 'YES'
);
PREPARE expert_index_p1_stmt FROM @expert_index_ddl;
EXECUTE expert_index_p1_stmt;
DEALLOCATE PREPARE expert_index_p1_stmt;

SET @expert_index_ddl = (
    SELECT IF(
        COUNT(*) = 1,
        'ALTER TABLE expert_index_sync_task MODIFY COLUMN target_type VARCHAR(16) NOT NULL COMMENT ''expert/profile/employment/project/education''',
        'SELECT 1'
    )
    FROM information_schema.COLUMNS
    WHERE TABLE_SCHEMA = DATABASE()
      AND TABLE_NAME = 'expert_index_sync_task'
      AND COLUMN_NAME = 'target_type'
      AND IS_NULLABLE = 'YES'
);
PREPARE expert_index_p1_stmt FROM @expert_index_ddl;
EXECUTE expert_index_p1_stmt;
DEALLOCATE PREPARE expert_index_p1_stmt;

SET @expert_index_ddl = (
    SELECT IF(COUNT(*) = 0,
        'ALTER TABLE expert_index_sync_task ADD UNIQUE KEY uk_expert_sync_task_dedupe (dedupe_key)',
        'SELECT 1')
    FROM information_schema.STATISTICS
    WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'expert_index_sync_task' AND INDEX_NAME = 'uk_expert_sync_task_dedupe'
);
PREPARE expert_index_p1_stmt FROM @expert_index_ddl;
EXECUTE expert_index_p1_stmt;
DEALLOCATE PREPARE expert_index_p1_stmt;

SET @expert_index_ddl = (
    SELECT IF(COUNT(*) = 0,
        'ALTER TABLE expert_index_sync_task ADD KEY idx_expert_sync_task_retry_queue (sync_status, next_retry_at, id)',
        'SELECT 1')
    FROM information_schema.STATISTICS
    WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'expert_index_sync_task' AND INDEX_NAME = 'idx_expert_sync_task_retry_queue'
);
PREPARE expert_index_p1_stmt FROM @expert_index_ddl;
EXECUTE expert_index_p1_stmt;
DEALLOCATE PREPARE expert_index_p1_stmt;

SET @expert_index_ddl = (
    SELECT IF(COUNT(*) = 0,
        'ALTER TABLE expert_index_sync_task ADD KEY idx_expert_sync_task_history (expert_id, created_at)',
        'SELECT 1')
    FROM information_schema.STATISTICS
    WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'expert_index_sync_task' AND INDEX_NAME = 'idx_expert_sync_task_history'
);
PREPARE expert_index_p1_stmt FROM @expert_index_ddl;
EXECUTE expert_index_p1_stmt;
DEALLOCATE PREPARE expert_index_p1_stmt;

SET @expert_index_ddl = (
    SELECT IF(COUNT(*) = 0,
        'ALTER TABLE expert_index_sync_task ADD KEY idx_expert_sync_task_source_locator (source_table, source_record_id, source_updated_at)',
        'SELECT 1')
    FROM information_schema.STATISTICS
    WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'expert_index_sync_task' AND INDEX_NAME = 'idx_expert_sync_task_source_locator'
);
PREPARE expert_index_p1_stmt FROM @expert_index_ddl;
EXECUTE expert_index_p1_stmt;
DEALLOCATE PREPARE expert_index_p1_stmt;

SET @expert_index_ddl = (
    SELECT IF(COUNT(*) = 0,
        'ALTER TABLE expert_index_sync_task ADD CONSTRAINT chk_expert_sync_task_event_source CHECK (event_source IN (''poll'', ''cdc'', ''reconcile'', ''replay''))',
        'SELECT 1')
    FROM information_schema.TABLE_CONSTRAINTS
    WHERE CONSTRAINT_SCHEMA = DATABASE() AND TABLE_NAME = 'expert_index_sync_task' AND CONSTRAINT_NAME = 'chk_expert_sync_task_event_source'
);
PREPARE expert_index_p1_stmt FROM @expert_index_ddl;
EXECUTE expert_index_p1_stmt;
DEALLOCATE PREPARE expert_index_p1_stmt;

SET @expert_index_ddl = (
    SELECT IF(COUNT(*) = 0,
        'ALTER TABLE expert_index_sync_task ADD CONSTRAINT chk_expert_sync_task_target CHECK (target_type IN (''expert'', ''profile'', ''employment'', ''project'', ''education''))',
        'SELECT 1')
    FROM information_schema.TABLE_CONSTRAINTS
    WHERE CONSTRAINT_SCHEMA = DATABASE() AND TABLE_NAME = 'expert_index_sync_task' AND CONSTRAINT_NAME = 'chk_expert_sync_task_target'
);
PREPARE expert_index_p1_stmt FROM @expert_index_ddl;
EXECUTE expert_index_p1_stmt;
DEALLOCATE PREPARE expert_index_p1_stmt;

SET @expert_index_ddl = (
    SELECT IF(COUNT(*) = 0,
        'ALTER TABLE expert_index_sync_task ADD CONSTRAINT chk_expert_sync_task_stage CHECK (processing_stage IN (''candidate'', ''source_read'', ''normalized'', ''chunk_built'', ''validated'', ''contract_ready'', ''handed_off'', ''completed''))',
        'SELECT 1')
    FROM information_schema.TABLE_CONSTRAINTS
    WHERE CONSTRAINT_SCHEMA = DATABASE() AND TABLE_NAME = 'expert_index_sync_task' AND CONSTRAINT_NAME = 'chk_expert_sync_task_stage'
);
PREPARE expert_index_p1_stmt FROM @expert_index_ddl;
EXECUTE expert_index_p1_stmt;
DEALLOCATE PREPARE expert_index_p1_stmt;

SET @expert_index_ddl = (
    SELECT IF(COUNT(*) = 0,
        'ALTER TABLE expert_index_sync_task ADD CONSTRAINT chk_expert_sync_task_attempt_lease CHECK ((attempt_token IS NULL AND lease_until IS NULL) OR (attempt_token IS NOT NULL AND lease_until IS NOT NULL))',
        'SELECT 1')
    FROM information_schema.TABLE_CONSTRAINTS
    WHERE CONSTRAINT_SCHEMA = DATABASE() AND TABLE_NAME = 'expert_index_sync_task' AND CONSTRAINT_NAME = 'chk_expert_sync_task_attempt_lease'
);
PREPARE expert_index_p1_stmt FROM @expert_index_ddl;
EXECUTE expert_index_p1_stmt;
DEALLOCATE PREPARE expert_index_p1_stmt;
