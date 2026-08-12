-- 来源溯源字段语义统一迁移。
--
-- 适用范围：只用于已经执行过旧版 001 的数据库，且只执行一次。
-- 新建数据库直接执行新版 001，不需要执行本脚本。
--
-- 新语义：
--   source_table    = 物理来源表名；
--   source_field    = 参与 Chunk 生成的源列名，使用 + 拼接；
--   source_field_id = 来源行主键 ID；
--   source_item_key = 稳定业务项键，继续用于 chunk_id 和 Same Segment。

ALTER TABLE expert_index_employment_stage
    ADD COLUMN source_field VARCHAR(512) NULL AFTER source_table,
    ADD COLUMN source_field_id VARCHAR(128) NULL AFTER source_field;

UPDATE expert_index_employment_stage
SET source_field = 'company_name+position_title+start_date+end_date+work_description+is_employed',
    source_field_id = source_record_id;

ALTER TABLE expert_index_employment_stage
    DROP INDEX uk_expert_emp_source,
    DROP INDEX idx_expert_emp_source_updated,
    DROP COLUMN source_record_id,
    MODIFY source_field VARCHAR(512) NOT NULL COMMENT '参与生成 Chunk 的源列名，+拼接',
    MODIFY source_field_id VARCHAR(128) NOT NULL COMMENT '来源行主键 ID',
    ADD UNIQUE KEY uk_expert_emp_source (source_table, source_field_id, source_item_key),
    ADD KEY idx_expert_emp_source_updated (source_updated_at, source_field_id);

ALTER TABLE expert_index_project_stage
    ADD COLUMN source_field VARCHAR(512) NULL AFTER source_table,
    ADD COLUMN source_field_id VARCHAR(128) NULL AFTER source_field;

UPDATE expert_index_project_stage
SET source_field = 'project_title+company_name+start_date+end_date+work_description',
    source_field_id = source_record_id;

ALTER TABLE expert_index_project_stage
    DROP INDEX uk_expert_project_source,
    DROP INDEX idx_expert_project_source_updated,
    DROP COLUMN source_record_id,
    MODIFY source_field VARCHAR(512) NOT NULL COMMENT '参与生成 Chunk 的源列名，+拼接',
    MODIFY source_field_id VARCHAR(128) NOT NULL COMMENT '来源行主键 ID',
    ADD UNIQUE KEY uk_expert_project_source (source_table, source_field_id, source_item_key),
    ADD KEY idx_expert_project_source_updated (source_updated_at, source_field_id);

-- expert_chunk 旧 source_field_id 保存的是“表名:主键”。产品新定义要求
-- source_field_id 直接保存主键，因此先用 source_record_id 覆盖回填。
UPDATE expert_chunk
SET source_field_id = source_record_id,
    source_field = CASE chunk_type
        WHEN 'profile_summary' THEN 'company+job_name+background'
        WHEN 'employment' THEN 'company_name+position_title+start_date+end_date+work_description+is_employed'
        WHEN 'project' THEN 'project_title+company_name+start_date+end_date+work_description'
        ELSE source_field
    END;

ALTER TABLE expert_chunk
    DROP INDEX idx_expert_chunk_source,
    DROP COLUMN source_record_id,
    MODIFY source_field VARCHAR(512) NOT NULL COMMENT '参与生成 Chunk 的源列名，+拼接',
    MODIFY source_field_id VARCHAR(128) NOT NULL COMMENT '来源行主键 ID',
    ADD KEY idx_expert_chunk_source (source_table, source_field_id, source_item_key);
