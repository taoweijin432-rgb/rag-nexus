-- V1.3：四表来源对应的 Stage 结构迁移脚本。
--
-- 适用范围：已执行新版 002 的现有测试/索引库。
-- 新建空库只执行最新的 001，不需要执行本脚本。
--
-- 注意：本文件会删除 Stage 中已被产品取消的 parse_status/project_role 列。
-- 执行前应先备份现有派生表；本项目不会自动运行它。文件名保留历史编号，
-- 当前版本不再迁移或写入 expert_chunk。

-- 1. Profile Stage 接收新增主档规范字段及结构化教育经历聚合结果。
ALTER TABLE expert_index_stage
    ADD COLUMN job_status_norm VARCHAR(32) NULL AFTER role_level,
    ADD COLUMN abroad_flag_norm TINYINT(1) NULL AFTER job_status_norm,
    ADD COLUMN major_names_json JSON NULL AFTER school_names_json,
    ADD COLUMN eligible_for_recommendation TINYINT(1) NULL AFTER work_years,
    ADD COLUMN verification_status VARCHAR(32) NOT NULL DEFAULT 'source_summary'
        AFTER eligible_for_recommendation;

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
