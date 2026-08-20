-- 专家索引任务状态表初始化脚本。
--
-- 重要边界：本脚本只能在派生索引库执行。演示环境没有单独的索引库配置时，CLI 会使用 DB_NAME；
-- 生产环境必须通过 INDEX_DB_NAME 指向独立 Schema。
-- 所有表均不存姓名、电话、邮箱、微信、身份证、银行卡、报价或附件 URL。

CREATE TABLE IF NOT EXISTS expert_index_batch (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    batch_no VARCHAR(64) NOT NULL,
    run_mode VARCHAR(16) NOT NULL COMMENT 'full/incremental/replay/single',
    -- data_version 属于加工批次，不进入下游内存 Contract。
    -- 先允许 NULL，保证第一阶段只升级表结构时，现有 single 命令仍能正常创建批次；
    -- 后续批量编排接入后，full 批次必须显式写入该值。
    data_version VARCHAR(64) NULL COMMENT '本批次使用的业务数据版本',
    status VARCHAR(16) NOT NULL COMMENT 'running/success/failed/partial',
    -- cursor_updated_at 为以后增量模式预留；第一版全量模式只使用 cursor_id。
    cursor_updated_at DATETIME(6) NULL,
    cursor_id BIGINT UNSIGNED NULL COMMENT '最后完成调度的专家 ID 游标',
    -- full 批次开始时固定 experts.id 的上界，防止运行期间新增专家让任务范围不断增长。
    upper_bound_id BIGINT UNSIGNED NULL COMMENT '全量批次启动时固定的最大专家 ID',
    -- candidate/success/failed/deactivated/rejected 均按“专家数”统计；
    -- extracted/chunk 按来源记录和 Chunk 行数统计；stage_count 仅为旧批次兼容字段，
    -- 纯内存标准化链路不再增加该值。
    candidate_count INT UNSIGNED NOT NULL DEFAULT 0,
    success_count INT UNSIGNED NOT NULL DEFAULT 0,
    failed_count INT UNSIGNED NOT NULL DEFAULT 0,
    deactivated_count INT UNSIGNED NOT NULL DEFAULT 0,
    extracted_count INT UNSIGNED NOT NULL DEFAULT 0,
    stage_count INT UNSIGNED NOT NULL DEFAULT 0 COMMENT '旧 Stage 批次兼容计数，新任务恒为 0',
    chunk_count INT UNSIGNED NOT NULL DEFAULT 0,
    rejected_count INT UNSIGNED NOT NULL DEFAULT 0,
    -- 以下字段累计下游已可靠返回的同步结果，用于断点续跑和批次历史查询。
    vector_sync_expert_count INT UNSIGNED NOT NULL DEFAULT 0
        COMMENT '已完成向量同步并返回可靠结果的专家数',
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
        COMMENT '成功删除的旧向量总数',
    vector_delete_not_found_count INT UNSIGNED NOT NULL DEFAULT 0
        COMMENT '请求删除但下游未找到的向量总数',
    error_summary JSON NULL,
    started_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    finished_at DATETIME(6) NULL,
    PRIMARY KEY (id),
    UNIQUE KEY uk_expert_index_batch_no (batch_no)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 每个物理来源表独立保存一个已经安全提交的增量扫描位置。
--
-- last_updated_at 与 last_id 组成复合游标：来源表 updated_at 当前只有秒级精度，
-- 同一秒内可能出现多条记录，因此不能只保存时间。Checkpoint 只表示该位置及之前的
-- 来源记录已经完成 Chunk/Metadata、必要的向量写入和旧向量清理；批次处理中间位置
-- 不得提前写入本表。last_batch_id 是逻辑关联，不设置物理外键，保持批次审计可独立维护。
CREATE TABLE IF NOT EXISTS expert_index_source_checkpoint (
    source_table VARCHAR(128) NOT NULL COMMENT '物理来源表名，每张来源表独立维护一行',
    last_updated_at DATETIME(6) NOT NULL COMMENT '最后安全提交的来源更新时间游标',
    last_id BIGINT UNSIGNED NOT NULL COMMENT 'last_updated_at 相同时最后安全提交的来源主键',
    last_batch_id BIGINT UNSIGNED NULL COMMENT '最后成功推进 Checkpoint 的构建批次 ID',
    last_success_time DATETIME(6) NULL COMMENT '最近一次安全提交 Checkpoint 的系统时间',
    lock_version BIGINT UNSIGNED NOT NULL DEFAULT 0
        COMMENT '乐观锁版本，防止旧批次覆盖较新的 Checkpoint',
    created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6)
        ON UPDATE CURRENT_TIMESTAMP(6),
    PRIMARY KEY (source_table),
    CONSTRAINT chk_expert_index_source_checkpoint_table CHECK (
        source_table IN (
            'experts',
            'experts_work_experience',
            'experts_project_experience',
            'experts_education_experience'
        )
    )
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 全量批次的专家级执行明细。
--
-- Pipeline 每发现一位候选专家，会为同一 (batch_id, expert_id) 创建或更新一行：
-- pending 表示等待处理；running 表示已经开始；success 表示正常产出；
-- rejected 表示专家存在但没有可生成内容；deactivated 表示专家已逻辑删除且历史
-- Chunk 已停用；failed 表示这一位专家发生异常。批量任务可通过本表继续处理其他专家，
-- 并在中断后只重试未完成或失败的专家。
--
-- 本表只保存计数、机器可读结果码和截断后的安全错误摘要，禁止保存专家背景、经历正文、
-- Chunk Text 或其他个人信息。batch_id 与 expert_index_batch.id 逻辑关联；项目现有派生表
-- 统一不设置物理外键，避免迁移顺序和批次审计记录清理互相阻塞。
CREATE TABLE IF NOT EXISTS expert_index_batch_item (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    batch_id BIGINT UNSIGNED NOT NULL COMMENT '逻辑关联 expert_index_batch.id',
    expert_id BIGINT UNSIGNED NOT NULL COMMENT '本明细处理的 experts.id',
    status VARCHAR(16) NOT NULL DEFAULT 'pending'
        COMMENT 'pending/running/success/rejected/deactivated/failed',
    attempt_count SMALLINT UNSIGNED NOT NULL DEFAULT 0 COMMENT '本批次内实际处理尝试次数',
    extracted_count INT UNSIGNED NOT NULL DEFAULT 0 COMMENT '该专家读取的四表来源记录数',
    stage_count INT UNSIGNED NOT NULL DEFAULT 0 COMMENT '旧 Stage 批次兼容计数，新任务恒为 0',
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
