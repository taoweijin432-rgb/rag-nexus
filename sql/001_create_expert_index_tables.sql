-- 专家索引派生表初始化脚本。
--
-- 重要边界：本脚本只能在派生索引库执行。演示环境没有单独的索引库配置时，CLI 会使用 DB_NAME；
-- 生产环境必须通过 INDEX_DB_NAME 指向独立 Schema。
-- 所有表均不存姓名、电话、邮箱、微信、身份证、银行卡、报价或附件 URL。

CREATE TABLE IF NOT EXISTS expert_index_batch (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    batch_no VARCHAR(64) NOT NULL,
    run_mode VARCHAR(16) NOT NULL COMMENT 'full/incremental/replay/single',
    -- data_version 属于加工批次与 Stage，不再进入最终 expert_chunk。
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
    -- extracted/stage/chunk 则保持现有含义，按来源记录、Stage、Chunk 行数统计。
    candidate_count INT UNSIGNED NOT NULL DEFAULT 0,
    success_count INT UNSIGNED NOT NULL DEFAULT 0,
    failed_count INT UNSIGNED NOT NULL DEFAULT 0,
    deactivated_count INT UNSIGNED NOT NULL DEFAULT 0,
    extracted_count INT UNSIGNED NOT NULL DEFAULT 0,
    stage_count INT UNSIGNED NOT NULL DEFAULT 0,
    chunk_count INT UNSIGNED NOT NULL DEFAULT 0,
    rejected_count INT UNSIGNED NOT NULL DEFAULT 0,
    error_summary JSON NULL,
    started_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    finished_at DATETIME(6) NULL,
    PRIMARY KEY (id),
    UNIQUE KEY uk_expert_index_batch_no (batch_no)
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

CREATE TABLE IF NOT EXISTS expert_index_stage (
    expert_id BIGINT UNSIGNED NOT NULL,
    serial_number VARCHAR(64) NULL,
    company_name_raw VARCHAR(512) NULL,
    company_name_norm VARCHAR(512) NULL,
    company_normalization_status VARCHAR(32) NOT NULL DEFAULT 'unknown',
    role_name_raw VARCHAR(256) NULL,
    role_code VARCHAR(128) NULL,
    role_level VARCHAR(64) NULL,
    job_status_norm VARCHAR(32) NULL,
    abroad_flag_norm TINYINT(1) NULL,
    background_clean MEDIUMTEXT NULL,
    school_names_json JSON NOT NULL,
    major_names_json JSON NOT NULL,
    work_years DECIMAL(5,2) NULL,
    eligible_for_recommendation TINYINT(1) NULL,
    verification_status VARCHAR(32) NOT NULL COMMENT 'doubtful_flag 映射结果',
    source_hash CHAR(64) NOT NULL,
    source_updated_at DATETIME(6) NULL,
    data_version VARCHAR(64) NOT NULL,
    stage_status VARCHAR(16) NOT NULL DEFAULT 'ready',
    reject_reason VARCHAR(512) NULL,
    build_batch_id BIGINT UNSIGNED NOT NULL,
    created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6)
        ON UPDATE CURRENT_TIMESTAMP(6),
    PRIMARY KEY (expert_id),
    KEY idx_expert_stage_batch (build_batch_id),
    KEY idx_expert_stage_source_updated (source_updated_at, expert_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS expert_index_employment_stage (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    expert_id BIGINT UNSIGNED NOT NULL,
    source_table VARCHAR(128) NOT NULL COMMENT '物理来源表名',
    source_field VARCHAR(512) NOT NULL COMMENT '参与生成 Chunk 的源列名，+拼接',
    source_field_id VARCHAR(128) NOT NULL COMMENT '来源行主键 ID',
    source_item_key VARCHAR(128) NOT NULL COMMENT '稳定业务项键',
    company_name_raw VARCHAR(512) NULL,
    company_name_norm VARCHAR(512) NULL,
    company_normalization_status VARCHAR(32) NOT NULL DEFAULT 'unknown',
    role_name_raw VARCHAR(256) NULL,
    role_code VARCHAR(128) NULL,
    role_level VARCHAR(64) NULL,
    start_month CHAR(6) NULL COMMENT '标准年月 YYYYMM',
    end_month CHAR(6) NULL COMMENT '标准年月 YYYYMM',
    is_current TINYINT(1) NULL,
    responsibilities_clean MEDIUMTEXT NULL,
    achievements_clean MEDIUMTEXT NULL,
    industry_codes_json JSON NOT NULL,
    topic_codes_json JSON NOT NULL,
    source_hash CHAR(64) NOT NULL,
    source_updated_at DATETIME(6) NULL,
    data_version VARCHAR(64) NOT NULL,
    build_batch_id BIGINT UNSIGNED NOT NULL,
    created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6)
        ON UPDATE CURRENT_TIMESTAMP(6),
    PRIMARY KEY (id),
    UNIQUE KEY uk_expert_emp_source (source_table, source_field_id, source_item_key),
    KEY idx_expert_emp_expert (expert_id),
    KEY idx_expert_emp_source_updated (source_updated_at, source_field_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS expert_index_project_stage (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    expert_id BIGINT UNSIGNED NOT NULL,
    source_table VARCHAR(128) NOT NULL COMMENT '物理来源表名',
    source_field VARCHAR(512) NOT NULL COMMENT '参与生成 Chunk 的源列名，+拼接',
    source_field_id VARCHAR(128) NOT NULL COMMENT '来源行主键 ID',
    source_item_key VARCHAR(128) NOT NULL COMMENT '稳定业务项键',
    project_name VARCHAR(512) NULL,
    company_or_client_raw VARCHAR(512) NULL,
    company_or_client_norm VARCHAR(512) NULL,
    company_normalization_status VARCHAR(32) NOT NULL DEFAULT 'unknown',
    start_month CHAR(6) NULL COMMENT '标准年月 YYYYMM',
    end_month CHAR(6) NULL COMMENT '标准年月 YYYYMM',
    objective_clean MEDIUMTEXT NULL,
    actions_clean MEDIUMTEXT NULL,
    results_clean MEDIUMTEXT NULL,
    industry_codes_json JSON NOT NULL,
    topic_codes_json JSON NOT NULL,
    source_hash CHAR(64) NOT NULL,
    source_updated_at DATETIME(6) NULL,
    data_version VARCHAR(64) NOT NULL,
    build_batch_id BIGINT UNSIGNED NOT NULL,
    created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6)
        ON UPDATE CURRENT_TIMESTAMP(6),
    PRIMARY KEY (id),
    UNIQUE KEY uk_expert_project_source (source_table, source_field_id, source_item_key),
    KEY idx_expert_project_expert (expert_id),
    KEY idx_expert_project_source_updated (source_updated_at, source_field_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS expert_chunk (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    vector_doc_id VARCHAR(256) NOT NULL COMMENT 'Milvus 唯一标识，不追加版本后缀',
    expert_id BIGINT UNSIGNED NOT NULL,
    serial_number VARCHAR(64) NULL,
    chunk_type VARCHAR(32) NOT NULL,
    chunk_subtype VARCHAR(32) NULL,
    chunk_index SMALLINT UNSIGNED NOT NULL DEFAULT 0,
    chunk_total SMALLINT UNSIGNED NOT NULL DEFAULT 1,
    chunk_text MEDIUMTEXT NOT NULL,
    char_count INT UNSIGNED NOT NULL,
    -- 以下为三种 Chunk Metadata 的全字段并集；不适用当前类型的字段保存 NULL/[]。
    source_table VARCHAR(256) NOT NULL COMMENT '来源表；聚合多个来源时使用 + 拼接',
    source_field VARCHAR(512) NOT NULL COMMENT '参与生成 Chunk 的源列名，+拼接',
    source_field_id VARCHAR(128) NOT NULL COMMENT '来源行主键 ID',
    source_version VARCHAR(64) NOT NULL,
    source_updated_at DATETIME(6) NULL,
    source_hash CHAR(64) NOT NULL,
    content_hash CHAR(64) NOT NULL,
    metadata_hash CHAR(64) NOT NULL COMMENT '稳定 Metadata 的 SHA-256',
    company_name_raw VARCHAR(512) NULL,
    company_name_norm VARCHAR(512) NULL,
    role_name_raw VARCHAR(256) NULL,
    role_code VARCHAR(128) NULL,
    role_level VARCHAR(64) NULL,
    job_status_norm VARCHAR(32) NULL,
    abroad_flag_norm TINYINT(1) NULL,
    work_years DECIMAL(5,2) NULL,
    project_name VARCHAR(512) NULL,
    start_month CHAR(6) NULL COMMENT '标准年月 YYYYMM',
    end_month CHAR(6) NULL COMMENT '标准年月 YYYYMM',
    is_current TINYINT(1) NULL,
    industry_codes JSON NOT NULL,
    function_codes JSON NOT NULL,
    topic_codes JSON NOT NULL,
    school_names_raw JSON NOT NULL,
    major_names_raw JSON NOT NULL,
    eligible_for_recommendation TINYINT(1) NULL COMMENT '仅 profile_summary 使用',
    origin_type VARCHAR(32) NOT NULL,
    verification_status VARCHAR(32) NOT NULL,
    weight_tier CHAR(1) NOT NULL,
    chunk_policy_version VARCHAR(32) NOT NULL,
    metadata_schema_version VARCHAR(32) NOT NULL,
    embedding_model_alias VARCHAR(128) NULL COMMENT '下游完成 Embedding 后回填',
    embedding_model_version VARCHAR(128) NULL COMMENT '下游完成 Embedding 后回填',
    indexed_at DATETIME(6) NULL COMMENT '下游写入 Milvus 成功后回填',
    is_active TINYINT(1) NOT NULL DEFAULT 1,
    build_batch_id BIGINT UNSIGNED NOT NULL,
    created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6)
        ON UPDATE CURRENT_TIMESTAMP(6),
    PRIMARY KEY (id),
    UNIQUE KEY uk_expert_chunk_vector_doc_id (vector_doc_id),
    KEY idx_expert_chunk_expert_active (expert_id, is_active),
    KEY idx_expert_chunk_type_active (chunk_type, is_active),
    KEY idx_expert_chunk_source (source_table, source_field_id),
    KEY idx_expert_chunk_content_hash (content_hash),
    KEY idx_expert_chunk_metadata_hash (metadata_hash),
    CONSTRAINT chk_expert_chunk_position CHECK (chunk_index < chunk_total),
    CONSTRAINT chk_expert_chunk_type CHECK (
        chunk_type IN ('profile_summary', 'employment', 'project')
    ),
    CONSTRAINT chk_expert_chunk_profile_eligibility CHECK (
        chunk_type = 'profile_summary' OR eligible_for_recommendation IS NULL
    )
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
