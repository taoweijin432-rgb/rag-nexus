-- 为现有派生索引库增加按物理来源表独立维护的安全 Checkpoint。
--
-- 已确认四张来源表均使用非空自增 id、非空且自动更新的 updated_at，并通过
-- deleted_at 表示逻辑删除。updated_at 当前为秒级 DATETIME，所以增量扫描必须按
-- (updated_at, id) 排序和翻页；last_id 用来避免同一秒内的数据重复或遗漏。
--
-- 本表只保存已经安全完成全部必要处理的数据位置。扫描中的临时游标、存在未决重试
-- 任务的批次位置都不能写入本表。该迁移只创建状态表，不初始化游标，也不修改来源库。

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
