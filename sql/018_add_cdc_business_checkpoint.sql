-- P7.7：保存已完成下游收敛的 Canal 业务位点。
--
-- 本表只保存 Index MySQL 的业务应用位点，不代替 Canal Server 的传输
-- Offset。Canal batch_id 是连接期间的确认句柄，不作为跨重启的 Checkpoint
-- 身份。迁移不修改四张 Source 表、expert_chunk 或向量表。
--
-- 前置：009 和 017 已部署。
-- 幂等性：新表使用 CREATE TABLE IF NOT EXISTS，可重复执行。
-- 回滚：先关闭 CDC 入口并停止 ACK；保留本表供恢复和审计。

CREATE TABLE IF NOT EXISTS expert_index_cdc_checkpoint (
    destination VARCHAR(128) NOT NULL COMMENT 'Canal destination，每个 destination 独立水位',
    binlog_server_id BIGINT UNSIGNED NOT NULL COMMENT '已应用位点的 MySQL server_id',
    binlog_file VARCHAR(128) NOT NULL COMMENT '已应用的 Binlog 文件',
    binlog_position BIGINT UNSIGNED NOT NULL COMMENT '已应用的 Binlog 文件内位置',
    last_local_batch_id BIGINT UNSIGNED NOT NULL COMMENT '最近提交的 Index 本地批次 ID',
    lock_version BIGINT UNSIGNED NOT NULL DEFAULT 0 COMMENT 'CAS 乐观锁版本',
    last_success_at DATETIME(6) NOT NULL COMMENT '最近一次成功提交的 Index MySQL 时间',
    created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6)
        ON UPDATE CURRENT_TIMESTAMP(6),
    PRIMARY KEY (destination),
    KEY idx_expert_cdc_checkpoint_position (
        binlog_server_id, binlog_file, binlog_position
    )
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
