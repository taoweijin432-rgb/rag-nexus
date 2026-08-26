-- 014：扩展 expert_index_sync_task 的 error_stage 取值集。
--
-- 背景：
--   sql/012 的 CREATE TABLE 把 chk_expert_sync_task_error_stage 固定为 9 个
--   失败阶段（source/parse/pii/chunk_builder/generation/validation/embedding/
--   milvus/checkpoint）。P5.3A 已在 expert_index/models.py 的
--   SOURCE_SYNC_ERROR_STAGES 中新增 4 个正式交付链编排阶段
--   （artifact/handoff/scope_consumer/receipt），若不同步 DDL，新值写入会被
--   CHECK 拒绝。本迁移在保留原有 9 个值的前提下补齐这 4 个值，使 DDL 与代码
--   模型一致。
--
-- 范围约束：
--   - 只修改 expert_index_sync_task 的 chk_expert_sync_task_error_stage CHECK；
--   - 不修改任何字段、索引或其他 CHECK/外键约束；
--   - 不创建或删除表，不写业务数据。
--
-- 幂等性：
--   - 通过 information_schema.CHECK_CONSTRAINTS 区分目标约束、旧约束和约束缺失；
--   - 已是目标约束时跳过；旧约束执行 DROP+ADD；约束缺失时直接 ADD；
--   - 若上次在 DROP 后、ADD 前中断，重跑会识别为“约束缺失”并补建；
--   - 表不存在（012 尚未部署）时下游 ALTER 会报错中止，符合“部署顺序
--     012 → 013 → 014 固定、失败应可见”的要求，本迁移不静默建表或兜底。
--
-- 部署顺序：012_complete_incremental_control_tables.sql
--          → 013_add_incremental_scan_lookup_index.sql
--          → 014_expand_expert_sync_task_error_stages.sql
-- 不要修改已提交的 sql/012；本迁移是独立、可重放的后置扩展。

-- 1) 读取约束存在性和目标版本状态。
SET @expert_index_p5_constraint_exists = (
    SELECT COUNT(*)
    FROM information_schema.CHECK_CONSTRAINTS cc
    INNER JOIN information_schema.TABLE_CONSTRAINTS tc
      ON tc.CONSTRAINT_SCHEMA = cc.CONSTRAINT_SCHEMA
     AND tc.CONSTRAINT_NAME = cc.CONSTRAINT_NAME
     AND tc.CONSTRAINT_TYPE = 'CHECK'
    WHERE cc.CONSTRAINT_SCHEMA = DATABASE()
      AND tc.TABLE_NAME = 'expert_index_sync_task'
      AND cc.CONSTRAINT_NAME = 'chk_expert_sync_task_error_stage'
);

SET @expert_index_p5_constraint_current = (
    SELECT
        CASE
            WHEN COUNT(*) = 1 AND MAX(CHECK_CLAUSE) LIKE '%receipt%' THEN 1
            ELSE 0
        END
    FROM information_schema.CHECK_CONSTRAINTS cc
    INNER JOIN information_schema.TABLE_CONSTRAINTS tc
      ON tc.CONSTRAINT_SCHEMA = cc.CONSTRAINT_SCHEMA
     AND tc.CONSTRAINT_NAME = cc.CONSTRAINT_NAME
     AND tc.CONSTRAINT_TYPE = 'CHECK'
    WHERE cc.CONSTRAINT_SCHEMA = DATABASE()
      AND tc.TABLE_NAME = 'expert_index_sync_task'
      AND cc.CONSTRAINT_NAME = 'chk_expert_sync_task_error_stage'
);

SET @expert_index_p5_need_drop = (
    @expert_index_p5_constraint_exists = 1
    AND @expert_index_p5_constraint_current = 0
);
SET @expert_index_p5_need_add = (
    @expert_index_p5_constraint_current = 0
);

-- 2) 卸载旧 CHECK（仅当需要升级时执行）。MySQL 8.0.16+ 使用 DROP CHECK 语法。
SET @expert_index_p5_ddl = (
    SELECT IF(
        (@expert_index_p5_need_drop = 1),
        'ALTER TABLE expert_index_sync_task DROP CHECK chk_expert_sync_task_error_stage',
        'SELECT 1'
    )
);
PREPARE expert_index_p5_stmt FROM @expert_index_p5_ddl;
EXECUTE expert_index_p5_stmt;
DEALLOCATE PREPARE expert_index_p5_stmt;

-- 3) 重新建立 CHECK，保留原有 9 个值并补齐 4 个 P5 阶段。
SET @expert_index_p5_ddl = (
    SELECT IF(
        (@expert_index_p5_need_add = 1),
        'ALTER TABLE expert_index_sync_task ADD CONSTRAINT chk_expert_sync_task_error_stage CHECK (error_stage IS NULL OR error_stage IN (''source'', ''parse'', ''pii'', ''chunk_builder'', ''generation'', ''validation'', ''embedding'', ''milvus'', ''checkpoint'', ''artifact'', ''handoff'', ''scope_consumer'', ''receipt''))',
        'SELECT 1'
    )
);
PREPARE expert_index_p5_stmt FROM @expert_index_p5_ddl;
EXECUTE expert_index_p5_stmt;
DEALLOCATE PREPARE expert_index_p5_stmt;
