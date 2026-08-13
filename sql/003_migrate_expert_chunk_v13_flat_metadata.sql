-- V1.3：四表来源与扁平 Metadata 迁移脚本。
--
-- 适用范围：已执行新版 002、且仍使用 metadata_json 的现有测试/索引库。
-- 新建空库只执行最新的 001，不需要执行本脚本。
--
-- 注意：本文件会删除 expert_chunk.metadata_json，以及 Stage 中已被产品取消的
-- parse_status/project_role 列。执行前应先备份现有派生表；本项目不会自动运行它。

-- 1. Profile Stage 接收新增主档规范字段及结构化教育经历聚合结果。
ALTER TABLE expert_index_stage
    ADD COLUMN job_status_norm VARCHAR(32) NULL AFTER role_level,
    ADD COLUMN abroad_flag_norm TINYINT(1) NULL AFTER job_status_norm,
    ADD COLUMN major_names_json JSON NULL AFTER school_names_json,
    ADD COLUMN eligible_for_recommendation TINYINT(1) NULL AFTER work_years;

UPDATE expert_index_stage
SET school_names_json = COALESCE(school_names_json, JSON_ARRAY()),
    major_names_json = COALESCE(major_names_json, JSON_ARRAY());

ALTER TABLE expert_index_stage
    MODIFY school_names_json JSON NOT NULL,
    MODIFY major_names_json JSON NOT NULL;

-- 2. 新日期规范改为 YYYYMM；Stage 不再保留解析状态和项目角色。
UPDATE expert_index_employment_stage
SET start_month = REPLACE(start_month, '-', ''),
    end_month = REPLACE(end_month, '-', '');

ALTER TABLE expert_index_employment_stage
    MODIFY start_month CHAR(6) NULL COMMENT '标准年月 YYYYMM',
    MODIFY end_month CHAR(6) NULL COMMENT '标准年月 YYYYMM',
    DROP COLUMN parse_status;

UPDATE expert_index_project_stage
SET start_month = REPLACE(start_month, '-', ''),
    end_month = REPLACE(end_month, '-', '');

ALTER TABLE expert_index_project_stage
    MODIFY start_month CHAR(6) NULL COMMENT '标准年月 YYYYMM',
    MODIFY end_month CHAR(6) NULL COMMENT '标准年月 YYYYMM',
    DROP COLUMN project_role,
    DROP COLUMN parse_status;

-- 3. 先增加三类 Metadata 的字段并集。迁移期允许 NULL，完成回填后再收紧必填列。
ALTER TABLE expert_chunk
    ADD COLUMN vector_doc_id VARCHAR(256) NULL AFTER chunk_id,
    ADD COLUMN metadata_hash CHAR(64) NULL AFTER content_hash,
    ADD COLUMN company_name_raw VARCHAR(512) NULL AFTER metadata_hash,
    ADD COLUMN role_name_raw VARCHAR(256) NULL AFTER company_name_norm,
    ADD COLUMN role_level VARCHAR(64) NULL AFTER role_code,
    ADD COLUMN job_status_norm VARCHAR(32) NULL AFTER role_level,
    ADD COLUMN abroad_flag_norm TINYINT(1) NULL AFTER job_status_norm,
    ADD COLUMN work_years DECIMAL(5,2) NULL AFTER abroad_flag_norm,
    ADD COLUMN project_name VARCHAR(512) NULL AFTER work_years,
    ADD COLUMN function_codes JSON NULL AFTER industry_codes_json,
    ADD COLUMN school_names_raw JSON NULL AFTER topic_codes_json,
    ADD COLUMN major_names_raw JSON NULL AFTER school_names_raw,
    ADD COLUMN eligible_for_recommendation TINYINT(1) NULL AFTER major_names_raw,
    ADD COLUMN embedding_model_alias VARCHAR(128) NULL AFTER data_version,
    ADD COLUMN embedding_model_version VARCHAR(128) NULL AFTER embedding_model_alias,
    ADD COLUMN indexed_at DATETIME(6) NULL AFTER embedding_model_version;

-- 4. 尽可能从旧 metadata_json 无损回填；缺失的数组字段统一为空数组。
UPDATE expert_chunk
SET vector_doc_id = COALESCE(
        JSON_UNQUOTE(JSON_EXTRACT(metadata_json, '$.vector_doc_id')),
        CONCAT('expert:', expert_id, ':', chunk_type, ':', source_field_id, ':', chunk_index)
    ),
    company_name_raw = JSON_UNQUOTE(JSON_EXTRACT(metadata_json, '$.company_name_raw')),
    role_name_raw = JSON_UNQUOTE(JSON_EXTRACT(metadata_json, '$.role_name_raw')),
    role_level = JSON_UNQUOTE(JSON_EXTRACT(metadata_json, '$.role_level')),
    project_name = JSON_UNQUOTE(JSON_EXTRACT(metadata_json, '$.project_name')),
    function_codes = COALESCE(JSON_EXTRACT(metadata_json, '$.function_codes'), JSON_ARRAY()),
    school_names_raw = COALESCE(JSON_EXTRACT(metadata_json, '$.school_names_raw'), JSON_ARRAY()),
    major_names_raw = COALESCE(JSON_EXTRACT(metadata_json, '$.major_names_raw'), JSON_ARRAY()),
    eligible_for_recommendation = JSON_EXTRACT(metadata_json, '$.eligible_for_recommendation'),
    metadata_hash = SHA2(CAST(metadata_json AS CHAR CHARACTER SET utf8mb4), 256),
    start_month = REPLACE(start_month, '-', ''),
    end_month = REPLACE(end_month, '-', '');

-- 旧列改成 V1.3 最终名称；现有行业/主题 JSON 内容可直接保留。
ALTER TABLE expert_chunk
    CHANGE COLUMN industry_codes_json industry_codes JSON NOT NULL,
    CHANGE COLUMN topic_codes_json topic_codes JSON NOT NULL,
    MODIFY function_codes JSON NOT NULL,
    MODIFY school_names_raw JSON NOT NULL,
    MODIFY major_names_raw JSON NOT NULL,
    MODIFY vector_doc_id VARCHAR(256) NOT NULL COMMENT 'Milvus 唯一标识，不追加版本后缀',
    MODIFY metadata_hash CHAR(64) NOT NULL COMMENT '稳定 Metadata 的 SHA-256',
    MODIFY start_month CHAR(6) NULL COMMENT '标准年月 YYYYMM',
    MODIFY end_month CHAR(6) NULL COMMENT '标准年月 YYYYMM';

-- 迁移生成的 metadata_hash 只用于保证历史行完整；下一次按专家重建时，程序会按
-- V1.3 稳定字段清单重新计算准确哈希，并覆盖对应行。
ALTER TABLE expert_chunk
    DROP COLUMN metadata_json,
    ADD UNIQUE KEY uk_expert_chunk_vector_doc_id (vector_doc_id),
    ADD KEY idx_expert_chunk_metadata_hash (metadata_hash),
    ADD CONSTRAINT chk_expert_chunk_profile_eligibility CHECK (
        chunk_type = 'profile_summary' OR eligible_for_recommendation IS NULL
    );
