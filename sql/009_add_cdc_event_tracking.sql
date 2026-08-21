-- 为派生索引库增加 Canal 行级变化的最小持久化任务表。
--
-- 消费者将事件可靠写入本表后才确认 Canal，业务任务再按 expert_id 回查来源表完整
-- 快照并执行 Chunk、Embedding 和向量同步。本表不保存完整 before/after 行、专家正文
-- 或 Chunk Text；唯一位点用于幂等接收重复投递。
--
-- 本迁移只创建 CDC 事件表，不消费 Canal、不初始化或推进 Checkpoint，也不修改来源库。

CREATE TABLE IF NOT EXISTS expert_index_cdc_event (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    destination VARCHAR(128) NOT NULL COMMENT 'Canal destination，如 company_expert',
    binlog_file VARCHAR(128) NOT NULL COMMENT '事件所在 MySQL Binlog 文件',
    binlog_position BIGINT UNSIGNED NOT NULL COMMENT '事件在 Binlog 文件中的偏移量',
    row_index INT UNSIGNED NOT NULL COMMENT '同一 Binlog 事件内从 0 开始的行序号',
    event_time DATETIME(6) NOT NULL COMMENT '来源 Binlog 事件发生时间',
    source_schema VARCHAR(128) NOT NULL COMMENT '发生变化的来源 Schema',
    source_table VARCHAR(128) NOT NULL COMMENT '发生变化的四张专家来源表之一',
    operation_type VARCHAR(16) NOT NULL COMMENT 'insert/update/delete',
    source_record_id BIGINT UNSIGNED NOT NULL COMMENT '发生变化的来源表主键 id',
    expert_id BIGINT UNSIGNED NULL COMMENT '受影响的 experts.id；无法映射时保留空值等待处置',
    status VARCHAR(16) NOT NULL DEFAULT 'pending'
        COMMENT 'pending/running/retry/success/dead_letter',
    attempt_count SMALLINT UNSIGNED NOT NULL DEFAULT 0 COMMENT '业务处理实际尝试次数',
    error_type VARCHAR(128) NULL COMMENT '安全异常类型，不保存来源行内容',
    error_message VARCHAR(512) NULL COMMENT '截断后的安全错误摘要，不保存 PII 或正文',
    processed_at DATETIME(6) NULL COMMENT '事件完成业务处理或进入 Dead Letter 的时间',
    created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6)
        ON UPDATE CURRENT_TIMESTAMP(6),
    PRIMARY KEY (id),
    UNIQUE KEY uk_expert_cdc_event_position (
        destination, binlog_file, binlog_position, row_index
    ),
    KEY idx_expert_cdc_event_queue (status, id),
    KEY idx_expert_cdc_event_expert (expert_id, event_time),
    CONSTRAINT chk_expert_cdc_event_table CHECK (
        source_table IN (
            'experts',
            'experts_work_experience',
            'experts_project_experience',
            'experts_education_experience'
        )
    ),
    CONSTRAINT chk_expert_cdc_event_operation CHECK (
        operation_type IN ('insert', 'update', 'delete')
    ),
    CONSTRAINT chk_expert_cdc_event_status CHECK (
        status IN ('pending', 'running', 'retry', 'success', 'dead_letter')
    )
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
