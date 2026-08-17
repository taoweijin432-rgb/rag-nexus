-- 向量化任务状态表（Task2 服务化 Step 1）。
--
-- ⚠️ 安全警告：
-- 本脚本创建向量化任务状态表，必须在本地开发数据库执行。
-- 执行前确认当前 USE 的数据库为 expert_dev 或本地开发库，
-- 禁止在公司业务数据库执行。
--
-- 验证方法：
--   SELECT DATABASE();  -- 确认返回 expert_dev 或本地开发库名
--
-- Task State 独立于 Milvus（AGENTS.md 架构约束）。
-- 使用独立数据库配置（TASK_STATE_MYSQL_* 环境变量），
-- 与 expert_chunk 读取配置（MYSQL_*）完全隔离。
--
-- 使用范围：
-- 1. 新建环境：先执行最新 001 创建 expert_chunk，再执行本文件创建 Task State 表；
-- 2. 已有环境：直接执行本文件（纯 CREATE TABLE IF NOT EXISTS，无 ALTER/DROP）。

-- ── 任务级表 ──
CREATE TABLE IF NOT EXISTS vectorization_task (
    task_id          CHAR(36)      NOT NULL COMMENT 'UUID v4',
    source           VARCHAR(20)   NOT NULL COMMENT 'full_batch / api_submit',
    mode             VARCHAR(20)   NOT NULL COMMENT '执行模式（Task2 固定 full）',
    status           VARCHAR(20)   NOT NULL DEFAULT 'created'
                     COMMENT 'created / running / completed / failed',
    total_count      INT UNSIGNED  NOT NULL DEFAULT 0  COMMENT '输入记录总数',
    success_count    INT UNSIGNED  NOT NULL DEFAULT 0  COMMENT '成功写入 Milvus 数量',
    skipped_count    INT UNSIGNED  NOT NULL DEFAULT 0  COMMENT 'Eligibility 跳过数量',
    failed_count     INT UNSIGNED  NOT NULL DEFAULT 0  COMMENT '处理失败数量',
    request_id       VARCHAR(64)   NULL COMMENT '幂等键，防止重复创建任务',
    error_message    VARCHAR(512)  NULL COMMENT '失败原因摘要，不含 PII/chunk_text',
    collection_name  VARCHAR(256)  NOT NULL COMMENT '写入的 Milvus Collection 名称',
    created_at       DATETIME(6)   NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    started_at       DATETIME(6)   NULL,
    finished_at      DATETIME(6)   NULL,
    PRIMARY KEY (task_id),
    UNIQUE KEY uk_vtask_request_id (request_id),
    KEY idx_vtask_status (status),
    KEY idx_vtask_source (source),
    KEY idx_vtask_created (created_at),
    CONSTRAINT chk_vtask_status CHECK (
        status IN ('created', 'running', 'completed', 'failed')
    ),
    CONSTRAINT chk_vtask_source CHECK (
        source IN ('full_batch', 'api_submit')
    )
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
  COMMENT='向量化任务表（一次批量向量化执行）';

-- ── 记录级表 ──
CREATE TABLE IF NOT EXISTS vectorization_task_item (
    task_id          CHAR(36)      NOT NULL COMMENT '关联 vectorization_task.task_id',
    vector_doc_id    VARCHAR(256)  NOT NULL COMMENT 'Contract 顶层身份键',
    status           VARCHAR(20)   NOT NULL DEFAULT 'pending'
                     COMMENT 'pending / success / skipped / failed',
    result           VARCHAR(128)  NULL COMMENT '处理结果详情（embedded / 跳过原因 / NULL）',
    error_message    VARCHAR(512)  NULL COMMENT '失败原因摘要，不含 PII/chunk_text',
    processed_at     DATETIME(6)   NULL COMMENT '到达终态时间',
    created_at       DATETIME(6)   NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    PRIMARY KEY (task_id, vector_doc_id),
    KEY idx_vtaskitem_status (task_id, status),
    CONSTRAINT chk_vtaskitem_status CHECK (
        status IN ('pending', 'success', 'skipped', 'failed')
    ),
    CONSTRAINT fk_vtaskitem_task FOREIGN KEY (task_id)
        REFERENCES vectorization_task (task_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
  COMMENT='向量化任务条目表（每条 vector_doc_id 的处理结果）';
