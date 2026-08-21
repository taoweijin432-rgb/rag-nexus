-- V15 增量向量化持久化层：Batch 与 Work Item 两张状态表。
--
-- 职责边界：
--   - 本迁移只创建 V15 增量向量化自己的任务状态表，不修改任何既有表；
--   - 不创建物理 delete 语义：disable 操作只把 is_active 置为 false，
--     向量与实体始终保留在 Milvus；
--   - Task State 独立于 Milvus 保存，Milvus 不是状态数据库。
--
-- 状态机（Work Item）：
--   created -> running -> committed   （Milvus 写入成功，不等待 HNSW 索引完成）
--   created -> running -> failed      （可重试的 Embedding / Milvus / 网络错误）
--   created -> rejected               （字段、模型空间或策略版本不符合）
--   running -> rejected               （处理中发现确定性校验失败，重试必然再失败）
--   running --(lease_until 过期)--> running（被下一个 Worker 重新认领，attempts + 1）
--
-- 幂等约束：
--   uk_v15_work_item_idem (batch_id, vector_doc_id_hash, operation)
--   同一批次内同一向量主键的同一操作只允许存在一条 Work Item；
--   vector_doc_id_hash = SHA-256(vector_doc_id)，用于规避 utf8mb4 长索引上限。
--
-- 上游关联：
--   v15_vectorization_batch.upstream_batch_id 只读引用上游触发批次
--   （如 expert_index_batch.batch_id）；保留本地自增 batch_id，
--   不建立跨模块外键。

CREATE TABLE IF NOT EXISTS v15_vectorization_batch (
    batch_id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    upstream_batch_id BIGINT UNSIGNED NULL
        COMMENT '上游触发批次 id（如 expert_index_batch.batch_id）；只读引用，不建外键',
    status VARCHAR(16) NOT NULL DEFAULT 'created'
        COMMENT 'created/running/committed/partial/failed',
    total_count INT UNSIGNED NOT NULL DEFAULT 0 COMMENT '计划处理的 Work Item 总数',
    committed_count INT UNSIGNED NOT NULL DEFAULT 0 COMMENT '已达到 committed 的数量',
    failed_count INT UNSIGNED NOT NULL DEFAULT 0 COMMENT '已达到 failed 的数量',
    rejected_count INT UNSIGNED NOT NULL DEFAULT 0 COMMENT '已达到 rejected 的数量',
    error_summary VARCHAR(1024) NULL COMMENT '截断后的安全错误摘要，不保存 PII 或正文',
    created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    started_at DATETIME(6) NULL COMMENT '最近一次进入 running 的时间',
    finished_at DATETIME(6) NULL COMMENT '到达终态（committed/partial/failed）的时间',
    updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6)
        ON UPDATE CURRENT_TIMESTAMP(6),
    PRIMARY KEY (batch_id),
    KEY idx_v15_batch_status (status, created_at),
    KEY idx_v15_batch_upstream (upstream_batch_id),
    CONSTRAINT chk_v15_batch_status CHECK (
        status IN ('created', 'running', 'committed', 'partial', 'failed')
    )
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS v15_vectorization_work_item (
    work_item_id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    batch_id BIGINT UNSIGNED NOT NULL COMMENT '逻辑关联 v15_vectorization_batch.batch_id',
    operation VARCHAR(16) NOT NULL COMMENT '只允许 upsert/metadata_update/disable，无物理 delete',
    vector_doc_id VARCHAR(2048) NOT NULL COMMENT 'V15 向量主键，与 Contract 上限 2048 字节一致',
    vector_doc_id_hash CHAR(64) NOT NULL COMMENT 'SHA-256(vector_doc_id)，幂等键与查找键',
    chunk_text TEXT NULL COMMENT 'Embedding 唯一输入；disable 项必须为 NULL',
    metadata_json JSON NULL COMMENT 'Contract metadata/source/versions 的 flat JSON',
    content_hash CHAR(64) NULL COMMENT '内容变化判断；disable 项为 NULL',
    metadata_hash CHAR(64) NULL COMMENT '非内容字段变化判断；metadata_update 必填',
    chunk_policy_version VARCHAR(32) NULL COMMENT 'Chunk 策略版本，模型空间校验依据之一',
    embedding_model_alias VARCHAR(128) NULL COMMENT '模型空间校验依据之一',
    embedding_model_version VARCHAR(64) NULL COMMENT '模型空间校验依据之一',
    is_active TINYINT(1) NOT NULL DEFAULT 1 COMMENT '本项应用后的向量生命周期标记；disable 为 0',
    expert_id BIGINT UNSIGNED NULL COMMENT '所属专家 id；disable 项可空',
    source_table VARCHAR(256) NULL COMMENT '四张专家来源表之一；disable 项可空',
    source_field_id VARCHAR(128) NULL COMMENT '来源字段定位 id；来源定位统一使用本字段命名',
    stale_vector_doc_ids JSON NULL COMMENT '本次操作需失效的旧 vector_doc_id 数组',
    status VARCHAR(16) NOT NULL DEFAULT 'created'
        COMMENT 'created/running/committed/failed/rejected',
    attempts INT UNSIGNED NOT NULL DEFAULT 0 COMMENT '被认领（含 lease 恢复重试）的次数',
    error_type VARCHAR(128) NULL COMMENT '安全异常类型，不保存来源内容',
    error_message VARCHAR(512) NULL COMMENT '截断后的安全错误摘要，不保存 PII 或正文',
    lease_until DATETIME(6) NULL COMMENT 'running 租约到期时间；过期后可被重新认领',
    processed_at DATETIME(6) NULL COMMENT '到达终态（committed/failed/rejected）的时间',
    created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6)
        ON UPDATE CURRENT_TIMESTAMP(6),
    PRIMARY KEY (work_item_id),
    UNIQUE KEY uk_v15_work_item_idem (batch_id, vector_doc_id_hash, operation),
    KEY idx_v15_work_item_queue (status, lease_until),
    KEY idx_v15_work_item_batch (batch_id, status),
    KEY idx_v15_work_item_vector (vector_doc_id_hash),
    KEY idx_v15_work_item_source (source_table, source_field_id),
    KEY idx_v15_work_item_expert (expert_id),
    CONSTRAINT chk_v15_work_item_operation CHECK (
        operation IN ('upsert', 'metadata_update', 'disable')
    ),
    CONSTRAINT chk_v15_work_item_status CHECK (
        status IN ('created', 'running', 'committed', 'failed', 'rejected')
    ),
    CONSTRAINT chk_v15_work_item_source_table CHECK (
        source_table IS NULL OR source_table IN (
            'experts',
            'experts_work_experience',
            'experts_project_experience',
            'experts_education_experience'
        )
    )
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
