-- 为现有派生索引库增加来源表增量批次的固定范围和临时扫描进度。
--
-- 本表与 expert_index_source_checkpoint 的含义不同：cursor_* 允许在扫描过程中推进，
-- 只用于中断恢复；它不证明 Chunk、Embedding、Milvus 写入或旧向量清理已经完成。
-- 只有该固定范围内的后续任务全部完成，才能单独提交安全 Checkpoint。
--
-- 该迁移只创建批次来源状态表，不创建任务、重试或 Dead Letter 结构，不初始化数据，
-- 也不修改业务来源库。

CREATE TABLE IF NOT EXISTS expert_index_batch_source (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    batch_id BIGINT UNSIGNED NOT NULL COMMENT '逻辑关联 expert_index_batch.id',
    source_table VARCHAR(128) NOT NULL COMMENT '本行负责扫描的物理来源表',
    start_updated_at DATETIME(6) NOT NULL COMMENT '批次启动时的安全 Checkpoint 时间',
    start_id BIGINT UNSIGNED NOT NULL COMMENT '批次启动时的安全 Checkpoint 来源主键',
    cursor_updated_at DATETIME(6) NOT NULL COMMENT '当前已扫描到的临时时间游标',
    cursor_id BIGINT UNSIGNED NOT NULL COMMENT '当前已扫描到的临时来源主键游标',
    upper_bound_updated_at DATETIME(6) NOT NULL COMMENT '批次启动时固定的扫描上界时间',
    upper_bound_id BIGINT UNSIGNED NOT NULL COMMENT '批次启动时固定的扫描上界来源主键',
    checkpoint_lock_version BIGINT UNSIGNED NOT NULL
        COMMENT '批次启动时读取的 Checkpoint 乐观锁版本',
    status VARCHAR(16) NOT NULL DEFAULT 'pending'
        COMMENT 'pending/running/success/failed',
    started_at DATETIME(6) NULL COMMENT '开始扫描该来源表的时间',
    finished_at DATETIME(6) NULL COMMENT '该来源表批次到达终态的时间',
    created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6)
        ON UPDATE CURRENT_TIMESTAMP(6),
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
