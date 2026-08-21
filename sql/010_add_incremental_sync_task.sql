-- 为现有派生索引库增加来源记录级增量同步任务表。
--
-- updated_at 扫描和 CDC 事件都可以生成本表任务。任务只保存来源定位、处理状态和
-- 安全错误摘要，不保存来源行、Chunk Text、Metadata 或 Embedding。
--
-- 本迁移只创建任务结构，不创建或执行任务，不处理重试，也不推进 Checkpoint。

CREATE TABLE IF NOT EXISTS expert_index_sync_task (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    batch_id BIGINT UNSIGNED NOT NULL COMMENT '逻辑关联 expert_index_batch.id',
    source_table VARCHAR(128) NOT NULL COMMENT '发生变化的四张专家来源表之一',
    source_record_id BIGINT UNSIGNED NOT NULL COMMENT '发生变化的来源表主键 id',
    expert_id BIGINT UNSIGNED NULL COMMENT '受影响的 experts.id；无法映射时等待后续处置',
    operation_type VARCHAR(16) NOT NULL COMMENT 'insert/update/delete',
    sync_status VARCHAR(16) NOT NULL DEFAULT 'pending'
        COMMENT 'pending/processing/success/failed/retrying/dead',
    retry_count SMALLINT UNSIGNED NOT NULL DEFAULT 0 COMMENT '自动重试次数，不包含首次处理',
    error_stage VARCHAR(32) NULL
        COMMENT 'source/parse/pii/chunk_builder/generation/validation/embedding/milvus/checkpoint',
    error_type VARCHAR(128) NULL COMMENT '安全异常类型，不保存来源内容',
    error_message VARCHAR(512) NULL COMMENT '截断后的安全错误摘要，不保存 PII 或正文',
    result_code VARCHAR(64) NULL COMMENT '机器结果码，如 no_chunkable_content',
    started_at DATETIME(6) NULL COMMENT '最近一次开始处理的时间',
    completed_at DATETIME(6) NULL COMMENT 'success 或 dead 的最终完成时间',
    created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    PRIMARY KEY (id),
    UNIQUE KEY uk_expert_sync_task_source (
        batch_id, source_table, source_record_id
    ),
    KEY idx_expert_sync_task_queue (sync_status, id),
    KEY idx_expert_sync_task_batch (batch_id, sync_status, id),
    KEY idx_expert_sync_task_expert (expert_id, id),
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
    CONSTRAINT chk_expert_sync_task_status CHECK (
        sync_status IN ('pending', 'processing', 'success', 'failed', 'retrying', 'dead')
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
