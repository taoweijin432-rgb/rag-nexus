-- Resume Index 控制表初始化脚本。
--
-- 强制边界：
-- 1. 本脚本只能由 resume_index.cli init-db 在 RESUME_INDEX_DB_* 指向的独立 Schema 执行。
-- 2. 本脚本不创建 Schema，不创建或修改 Resume 主表及三张 resume_* Source 表，
--    也不包含任何来源正文。
-- 3. 为复用现有 Repository，控制表沿用内部 expert_index_* 表名；由于 Resume Index
--    使用独立 Schema，不会与 Expert 控制表混用。
-- 4. 控制表中的 expert_id 是现有内部技术字段，在 Resume 链路中固定承载 resume_id。
-- 5. CDC 控制表只保存事件位点、状态和任务关联，不保存来源行正文；CDC 默认关闭。

-- 构建批次主表：保存运行边界、进度和下游动作计数，不保存 Resume 正文。
CREATE TABLE IF NOT EXISTS expert_index_batch (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    batch_no VARCHAR(64) NOT NULL,
    run_mode VARCHAR(16) NOT NULL COMMENT 'full/incremental/replay/single',
    data_version VARCHAR(64) NULL COMMENT '本批次使用的业务数据版本',
    status VARCHAR(16) NOT NULL COMMENT 'running/success/failed/partial',
    cursor_updated_at DATETIME(6) NULL,
    cursor_id BIGINT UNSIGNED NULL COMMENT '最后完成调度的 resume.id 物理游标',
    upper_bound_id BIGINT UNSIGNED NULL COMMENT '全量启动时固定的 resume.id 上界',
    candidate_count INT UNSIGNED NOT NULL DEFAULT 0,
    success_count INT UNSIGNED NOT NULL DEFAULT 0,
    failed_count INT UNSIGNED NOT NULL DEFAULT 0,
    deactivated_count INT UNSIGNED NOT NULL DEFAULT 0,
    extracted_count INT UNSIGNED NOT NULL DEFAULT 0,
    stage_count INT UNSIGNED NOT NULL DEFAULT 0 COMMENT '旧 Stage 兼容计数，新任务恒为 0',
    chunk_count INT UNSIGNED NOT NULL DEFAULT 0,
    rejected_count INT UNSIGNED NOT NULL DEFAULT 0,
    vector_sync_expert_count INT UNSIGNED NOT NULL DEFAULT 0
        COMMENT '已完成向量同步并返回可靠结果的 Resume 人物数',
    vector_current_count INT UNSIGNED NOT NULL DEFAULT 0
        COMMENT '上游提交的当前 Chunk 总数',
    vector_eligible_count INT UNSIGNED NOT NULL DEFAULT 0
        COMMENT '满足保留条件的 Chunk 总数',
    vector_embedded_count INT UNSIGNED NOT NULL DEFAULT 0
        COMMENT '实际重新向量化的 Chunk 总数',
    vector_metadata_updated_count INT UNSIGNED NOT NULL DEFAULT 0
        COMMENT '只更新 Metadata 的 Chunk 总数',
    vector_unchanged_count INT UNSIGNED NOT NULL DEFAULT 0
        COMMENT '无需写入的未变化 Chunk 总数',
    vector_deleted_count INT UNSIGNED NOT NULL DEFAULT 0
        COMMENT '成功失效的旧向量总数',
    vector_delete_not_found_count INT UNSIGNED NOT NULL DEFAULT 0
        COMMENT '请求失效但下游未找到的向量总数',
    error_summary JSON NULL,
    started_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    finished_at DATETIME(6) NULL,
    PRIMARY KEY (id),
    UNIQUE KEY uk_expert_index_batch_no (batch_no)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 四张物理来源表分别维护已经安全提交的 (updated_at, id) Checkpoint。
CREATE TABLE IF NOT EXISTS expert_index_source_checkpoint (
    source_table VARCHAR(128) NOT NULL COMMENT '四张 Resume 物理来源表之一',
    last_updated_at DATETIME(6) NOT NULL COMMENT '最后安全提交的来源更新时间游标',
    last_id BIGINT UNSIGNED NOT NULL COMMENT '同一更新时间内最后安全提交的来源主键',
    last_batch_id BIGINT UNSIGNED NULL COMMENT '最后推进 Checkpoint 的构建批次 ID',
    last_success_time DATETIME(6) NULL COMMENT '最近一次安全提交时间',
    lock_version BIGINT UNSIGNED NOT NULL DEFAULT 0 COMMENT 'Checkpoint 乐观锁版本',
    lease_owner VARCHAR(128) NULL COMMENT '当前扫描租约持有者',
    lease_until DATETIME(6) NULL COMMENT '扫描租约到期时间',
    created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6)
        ON UPDATE CURRENT_TIMESTAMP(6),
    PRIMARY KEY (source_table),
    KEY idx_expert_source_checkpoint_lease (lease_until, source_table),
    CONSTRAINT chk_expert_index_source_checkpoint_table CHECK (
        source_table IN (
            'resume',
            'resume_work_experience',
            'resume_project_experience',
            'resume_education_experience'
        )
    ),
    CONSTRAINT chk_expert_source_checkpoint_lease CHECK (
        (lease_owner IS NULL AND lease_until IS NULL)
        OR (lease_owner IS NOT NULL AND lease_until IS NOT NULL)
    )
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 一个批次对一张来源表的固定范围、临时游标和任务完成计数。
CREATE TABLE IF NOT EXISTS expert_index_batch_source (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    batch_id BIGINT UNSIGNED NOT NULL COMMENT '逻辑关联 expert_index_batch.id',
    source_table VARCHAR(128) NOT NULL COMMENT '本行负责扫描的 Resume 来源表',
    start_updated_at DATETIME(6) NOT NULL COMMENT '批次启动时的安全 Checkpoint 时间',
    start_id BIGINT UNSIGNED NOT NULL COMMENT '批次启动时的安全 Checkpoint 来源主键',
    cursor_updated_at DATETIME(6) NOT NULL COMMENT '当前已扫描到的临时时间游标',
    cursor_id BIGINT UNSIGNED NOT NULL COMMENT '当前已扫描到的临时来源主键',
    upper_bound_updated_at DATETIME(6) NOT NULL COMMENT '批次启动时固定的上界时间',
    upper_bound_id BIGINT UNSIGNED NOT NULL COMMENT '批次启动时固定的上界来源主键',
    checkpoint_lock_version BIGINT UNSIGNED NOT NULL
        COMMENT '批次启动时读取的 Checkpoint 乐观锁版本',
    status VARCHAR(16) NOT NULL DEFAULT 'pending'
        COMMENT 'pending/running/success/failed',
    candidate_count INT UNSIGNED NOT NULL DEFAULT 0 COMMENT '该范围发现的候选记录数',
    success_count INT UNSIGNED NOT NULL DEFAULT 0 COMMENT '该范围已完成的任务数',
    failed_count INT UNSIGNED NOT NULL DEFAULT 0 COMMENT '该范围失败或进入 DLQ 的任务数',
    skipped_count INT UNSIGNED NOT NULL DEFAULT 0 COMMENT '该范围无需下游动作的任务数',
    started_at DATETIME(6) NULL,
    finished_at DATETIME(6) NULL,
    created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6)
        ON UPDATE CURRENT_TIMESTAMP(6),
    PRIMARY KEY (id),
    UNIQUE KEY uk_expert_batch_source (batch_id, source_table),
    KEY idx_expert_batch_source_status (batch_id, status),
    KEY idx_expert_batch_source_claim (source_table, status, id),
    CONSTRAINT chk_expert_batch_source_table CHECK (
        source_table IN (
            'resume',
            'resume_work_experience',
            'resume_project_experience',
            'resume_education_experience'
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

-- Canal 交付给 Resume 处理链的最小事件记录。先幂等持久化，业务任务按 resume_id
-- 回查四张来源表的当前快照；本表不保存完整 before/after、简历正文或 Chunk Text。
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

-- 已完成下游收敛的 Resume Canal 业务位点。该水位不代替 Canal Server Offset，
-- Canal batch_id 也不作为跨重启身份。
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

-- 一条来源记录的标准处理任务。expert_id 在本 Schema 中固定表示 resume_id。
CREATE TABLE IF NOT EXISTS expert_index_sync_task (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    dedupe_key CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL
        COMMENT '按冻结公式生成的 SHA-256 跨批次幂等键',
    batch_id BIGINT UNSIGNED NOT NULL COMMENT '逻辑关联 expert_index_batch.id',
    event_source VARCHAR(16) NOT NULL DEFAULT 'poll'
        COMMENT 'poll/cdc/reconcile/replay',
    source_table VARCHAR(128) NOT NULL COMMENT '发生变化的 Resume 来源表',
    source_record_id BIGINT UNSIGNED NOT NULL COMMENT '来源表物理主键 id',
    source_updated_at DATETIME(6) NULL COMMENT '扫描候选的来源 updated_at',
    expert_id BIGINT UNSIGNED NULL COMMENT '内部兼容字段，Resume 链路中固定承载 resume_id',
    operation_type VARCHAR(16) NOT NULL COMMENT 'insert/update/delete',
    target_type VARCHAR(16) NOT NULL
        COMMENT 'expert/profile/employment/project/education',
    processing_stage VARCHAR(32) NOT NULL DEFAULT 'candidate'
        COMMENT 'candidate/source_read/normalized/chunk_built/validated/contract_ready/handed_off/completed',
    sync_status VARCHAR(16) NOT NULL DEFAULT 'pending'
        COMMENT 'pending/processing/success/failed/retrying/dead',
    retry_count SMALLINT UNSIGNED NOT NULL DEFAULT 0 COMMENT '自动重试次数，不含首次处理',
    attempt_token VARCHAR(64) NULL COMMENT '当前认领尝试令牌，防止过期 Worker 写回',
    lease_until DATETIME(6) NULL COMMENT '当前任务租约到期时间',
    next_retry_at DATETIME(6) NULL COMMENT '失败任务下一次允许认领的时间',
    policy_version VARCHAR(32) NULL COMMENT '产生幂等键和动作计划时使用的策略版本',
    error_stage VARCHAR(32) NULL COMMENT '失败所在的标准处理阶段',
    error_code VARCHAR(64) NULL COMMENT '稳定机器错误码，不保存来源内容',
    error_type VARCHAR(128) NULL COMMENT '安全异常类型，不保存来源内容',
    error_message VARCHAR(512) NULL COMMENT '截断后的安全错误摘要，不保存 PII 或正文',
    result_code VARCHAR(64) NULL COMMENT '机器结果码，如 no_chunkable_content',
    started_at DATETIME(6) NULL,
    completed_at DATETIME(6) NULL,
    created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6)
        ON UPDATE CURRENT_TIMESTAMP(6),
    PRIMARY KEY (id),
    UNIQUE KEY uk_expert_sync_task_dedupe (dedupe_key),
    UNIQUE KEY uk_expert_sync_task_source (batch_id, source_table, source_record_id),
    KEY idx_expert_sync_task_queue (sync_status, id),
    KEY idx_expert_sync_task_retry_queue (sync_status, next_retry_at, id),
    KEY idx_expert_sync_task_batch (batch_id, sync_status, id),
    KEY idx_expert_sync_task_expert (expert_id, id),
    KEY idx_expert_sync_task_history (expert_id, created_at),
    KEY idx_expert_sync_task_source_locator (
        source_table, source_record_id, source_updated_at
    ),
    CONSTRAINT chk_expert_sync_task_table CHECK (
        source_table IN (
            'resume',
            'resume_work_experience',
            'resume_project_experience',
            'resume_education_experience'
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
            'checkpoint',
            'artifact',
            'handoff',
            'scope_consumer',
            'receipt'
        )
    )
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 任务跨阶段恢复使用的受控 Contract 证据，不存储向量或来源事实。
CREATE TABLE IF NOT EXISTS expert_index_sync_artifact (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    task_id BIGINT UNSIGNED NOT NULL COMMENT '逻辑关联 expert_index_sync_task.id',
    artifact_type VARCHAR(32) NOT NULL COMMENT 'contract_payload/contract_reference',
    artifact_version VARCHAR(32) NOT NULL COMMENT '恢复 Payload 的 Contract 版本',
    payload_json JSON NULL COMMENT '通过白名单和 PII 校验的受控 Contract Payload',
    payload_reference VARCHAR(512) NULL COMMENT '受控 Contract Payload 引用',
    payload_hash CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL
        COMMENT 'Payload 或引用的 SHA-256 完整性摘要',
    expires_at DATETIME(6) NULL,
    created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6)
        ON UPDATE CURRENT_TIMESTAMP(6),
    PRIMARY KEY (id),
    UNIQUE KEY uk_expert_sync_artifact_version (
        task_id, artifact_type, artifact_version
    ),
    KEY idx_expert_sync_artifact_expiry (expires_at, id),
    CONSTRAINT chk_expert_sync_artifact_type CHECK (
        artifact_type IN ('contract_payload', 'contract_reference')
    ),
    CONSTRAINT chk_expert_sync_artifact_payload CHECK (
        (payload_json IS NOT NULL AND payload_reference IS NULL)
        OR (payload_json IS NULL AND payload_reference IS NOT NULL)
    )
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 每次失败 attempt 保留一条 Dead Letter 历史及受控处置审计。
CREATE TABLE IF NOT EXISTS expert_index_sync_dead_letter (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    task_id BIGINT UNSIGNED NOT NULL COMMENT '逻辑关联 expert_index_sync_task.id',
    batch_id BIGINT UNSIGNED NOT NULL COMMENT '冗余批次 ID，便于审计和补偿',
    source_table VARCHAR(128) NOT NULL COMMENT '发生失败的 Resume 来源表',
    source_record_id BIGINT UNSIGNED NOT NULL COMMENT '来源表物理主键 id',
    expert_id BIGINT UNSIGNED NULL COMMENT '内部兼容字段，Resume 链路中固定承载 resume_id',
    failure_stage VARCHAR(32) NOT NULL COMMENT '进入 Dead Letter 时的处理阶段',
    reason_code VARCHAR(64) NOT NULL COMMENT '稳定机器失败码',
    reason_message VARCHAR(512) NULL COMMENT '截断后的安全失败摘要，不保存 PII 或正文',
    retry_count SMALLINT UNSIGNED NOT NULL COMMENT '进入 Dead Letter 时的自动重试次数',
    artifact_id BIGINT UNSIGNED NULL COMMENT '逻辑关联受控恢复证据',
    resolution_status VARCHAR(16) NOT NULL DEFAULT 'pending'
        COMMENT 'pending/requeued/resolved/ignored',
    resolution_note VARCHAR(512) NULL COMMENT '截断后的安全处置说明',
    resolution_operator VARCHAR(128) NULL COMMENT '处置操作者安全标识',
    resolution_request_id VARCHAR(128) NULL COMMENT '处置请求幂等标识',
    resolution_version INT UNSIGNED NOT NULL DEFAULT 0 COMMENT '处置状态乐观锁版本',
    resolved_at DATETIME(6) NULL,
    created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6)
        ON UPDATE CURRENT_TIMESTAMP(6),
    PRIMARY KEY (id),
    UNIQUE KEY uk_expert_sync_dead_letter_attempt (task_id, retry_count),
    KEY idx_expert_sync_dead_letter_resolution (resolution_status, created_at),
    KEY idx_expert_sync_dead_letter_batch (batch_id, id),
    KEY idx_expert_sync_dead_letter_expert (expert_id, id),
    CONSTRAINT chk_expert_sync_dead_letter_table CHECK (
        source_table IN (
            'resume',
            'resume_work_experience',
            'resume_project_experience',
            'resume_education_experience'
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

-- 全量批次的人物级执行明细。expert_id 在本 Schema 中固定表示 resume_id。
CREATE TABLE IF NOT EXISTS expert_index_batch_item (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    batch_id BIGINT UNSIGNED NOT NULL COMMENT '逻辑关联 expert_index_batch.id',
    expert_id BIGINT UNSIGNED NOT NULL COMMENT '内部兼容字段，固定承载 resume_id',
    status VARCHAR(16) NOT NULL DEFAULT 'pending'
        COMMENT 'pending/running/success/rejected/deactivated/failed',
    attempt_count SMALLINT UNSIGNED NOT NULL DEFAULT 0,
    extracted_count INT UNSIGNED NOT NULL DEFAULT 0 COMMENT '读取的四表来源记录数',
    stage_count INT UNSIGNED NOT NULL DEFAULT 0 COMMENT '旧 Stage 兼容计数，新任务恒为 0',
    chunk_count INT UNSIGNED NOT NULL DEFAULT 0 COMMENT '本轮生成的 Chunk 数',
    result_code VARCHAR(64) NULL COMMENT '安全机器结果码',
    error_type VARCHAR(128) NULL COMMENT '异常类型，不保存来源正文',
    error_message VARCHAR(512) NULL COMMENT '截断后的安全错误摘要',
    started_at DATETIME(6) NULL,
    finished_at DATETIME(6) NULL,
    created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6)
        ON UPDATE CURRENT_TIMESTAMP(6),
    PRIMARY KEY (id),
    UNIQUE KEY uk_expert_batch_item (batch_id, expert_id),
    KEY idx_expert_batch_item_status (batch_id, status),
    KEY idx_expert_batch_item_history (expert_id, batch_id),
    CONSTRAINT chk_expert_batch_item_status CHECK (
        status IN ('pending', 'running', 'success', 'rejected', 'deactivated', 'failed')
    )
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
