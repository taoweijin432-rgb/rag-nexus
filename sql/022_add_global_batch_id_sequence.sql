-- 为全量、单条、Poll 和 CDC 建立统一 Batch ID 序列。
--
-- 历史 CDC 本地批次号没有写入 expert_index_batch，自增 ID 可能在未来再次使用同一
-- 数字。本迁移将所有已持久化批次引用的最大值作为序列水位，后续四类入口只能通过
-- 该序列原子分配编号。迁移可重复执行，只允许抬高水位，绝不回退。

CREATE TABLE IF NOT EXISTS expert_index_batch_id_sequence (
    sequence_name VARCHAR(64) NOT NULL COMMENT '批次序列名称，当前固定为 global',
    current_value BIGINT UNSIGNED NOT NULL COMMENT '最后一次已分配的全局 Batch ID',
    created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6)
        ON UPDATE CURRENT_TIMESTAMP(6),
    PRIMARY KEY (sequence_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

INSERT INTO expert_index_batch_id_sequence(sequence_name, current_value)
SELECT
    'global',
    GREATEST(
        COALESCE((SELECT MAX(id) FROM expert_index_batch), 0),
        COALESCE((SELECT MAX(batch_id) FROM expert_index_batch_source), 0),
        COALESCE((SELECT MAX(batch_id) FROM expert_index_batch_item), 0),
        COALESCE((SELECT MAX(batch_id) FROM expert_index_sync_task), 0),
        COALESCE((SELECT MAX(batch_id) FROM expert_index_sync_dead_letter), 0),
        COALESCE((SELECT MAX(last_batch_id) FROM expert_index_source_checkpoint), 0),
        COALESCE((SELECT MAX(local_batch_id) FROM expert_index_cdc_event), 0),
        COALESCE((SELECT MAX(last_local_batch_id) FROM expert_index_cdc_checkpoint), 0)
    )
ON DUPLICATE KEY UPDATE
    current_value=GREATEST(current_value, VALUES(current_value));
