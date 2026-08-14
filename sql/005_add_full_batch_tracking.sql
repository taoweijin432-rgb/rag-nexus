-- 为现有派生索引库增加第一版全量 Chunk 加工所需的批次跟踪结构。
--
-- 使用范围：
-- 1. 已经执行过旧版 001、数据库中已存在 expert_index_batch 的环境执行本迁移；
-- 2. 新建环境直接执行最新 001，不需要再执行本文件；
-- 3. 本迁移只增加兼容列和新表，不删除或改写现有批次、Stage、Chunk 数据；
-- 4. 执行本迁移后，Repository/Pipeline 才能运行 build-batch 并写入这些字段。

ALTER TABLE expert_index_batch
    -- data_version 继续属于批次和 Stage，不重新加入最终 expert_chunk。
    -- 允许 NULL 是为了让迁移后的现有 single 命令保持兼容；后续 full 编排必须写入。
    ADD COLUMN data_version VARCHAR(64) NULL
        COMMENT '本批次使用的业务数据版本' AFTER run_mode,
    -- 全量模式使用 cursor_id 断点续跑，并在启动时固定 upper_bound_id。
    MODIFY COLUMN cursor_id BIGINT UNSIGNED NULL
        COMMENT '最后完成调度的专家 ID 游标',
    ADD COLUMN upper_bound_id BIGINT UNSIGNED NULL
        COMMENT '全量批次启动时固定的最大专家 ID' AFTER cursor_id,
    -- 以下五列按专家计数；原有 extracted/stage/chunk 继续按数据行计数。
    ADD COLUMN candidate_count INT UNSIGNED NOT NULL DEFAULT 0 AFTER upper_bound_id,
    ADD COLUMN success_count INT UNSIGNED NOT NULL DEFAULT 0 AFTER candidate_count,
    ADD COLUMN failed_count INT UNSIGNED NOT NULL DEFAULT 0 AFTER success_count,
    ADD COLUMN deactivated_count INT UNSIGNED NOT NULL DEFAULT 0 AFTER failed_count;

-- 每个候选专家在一个批次中最多一行。批量编排通过该唯一键幂等创建明细，
-- 分别记录正常成功、无内容拒绝、逻辑删除停用和处理失败，并据此汇总 partial 批次。
CREATE TABLE IF NOT EXISTS expert_index_batch_item (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    batch_id BIGINT UNSIGNED NOT NULL COMMENT '逻辑关联 expert_index_batch.id',
    expert_id BIGINT UNSIGNED NOT NULL COMMENT '本明细处理的 experts.id',
    status VARCHAR(16) NOT NULL DEFAULT 'pending'
        COMMENT 'pending/running/success/rejected/deactivated/failed',
    attempt_count SMALLINT UNSIGNED NOT NULL DEFAULT 0 COMMENT '本批次内实际处理尝试次数',
    extracted_count INT UNSIGNED NOT NULL DEFAULT 0 COMMENT '该专家读取的四表来源记录数',
    stage_count INT UNSIGNED NOT NULL DEFAULT 0 COMMENT '该专家成功保存的 Stage 行数',
    chunk_count INT UNSIGNED NOT NULL DEFAULT 0 COMMENT '该专家本轮生成的 Chunk 行数',
    result_code VARCHAR(64) NULL
        COMMENT '安全机器码，如 no_chunkable_content/expert_deleted',
    error_type VARCHAR(128) NULL COMMENT '异常类型，不保存源数据正文',
    error_message VARCHAR(512) NULL COMMENT '截断后的安全错误摘要，不保存 PII 或 Chunk Text',
    started_at DATETIME(6) NULL COMMENT '该专家本轮首次开始处理的时间',
    finished_at DATETIME(6) NULL COMMENT '该专家到达终态的时间',
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
