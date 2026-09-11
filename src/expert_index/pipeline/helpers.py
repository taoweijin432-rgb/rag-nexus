"""构建编排使用的校验、映射和批次编号函数。"""

from __future__ import annotations

from datetime import datetime

from ..models import ExpertCandidate
from ..repositories import FullBatchCheckpoint
from .results import VectorSyncSummary


def _checkpoint_vector_sync(
    checkpoint: FullBatchCheckpoint,
) -> VectorSyncSummary:
    """把批次检查点中的持久化计数恢复为同步摘要。"""

    return VectorSyncSummary(
        expert_count=checkpoint.vector_sync_expert_count,
        current_count=checkpoint.vector_current_count,
        eligible_count=checkpoint.vector_eligible_count,
        embedded_count=checkpoint.vector_embedded_count,
        metadata_updated_count=checkpoint.vector_metadata_updated_count,
        unchanged_count=checkpoint.vector_unchanged_count,
        deleted_count=checkpoint.vector_deleted_count,
        delete_not_found_count=checkpoint.vector_delete_not_found_count,
    )


def _vector_sync_persistence_fields(summary: VectorSyncSummary) -> dict[str, int]:
    """把内存摘要映射为批次主表对应的数量参数。"""

    return {
        "vector_sync_expert_count": summary.expert_count,
        "vector_current_count": summary.current_count,
        "vector_eligible_count": summary.eligible_count,
        "vector_embedded_count": summary.embedded_count,
        "vector_metadata_updated_count": summary.metadata_updated_count,
        "vector_unchanged_count": summary.unchanged_count,
        "vector_deleted_count": summary.deleted_count,
        "vector_delete_not_found_count": summary.delete_not_found_count,
    }


def _validate_data_version(data_version: str) -> None:
    """拒绝空数据版本，避免创建无法追踪来源版本的构建批次。"""

    if not data_version.strip():
        raise ValueError("data_version 不能为空")


def _validate_expert_candidate_page(
    candidates: list[ExpertCandidate],
    *,
    after_id: int,
    upper_bound_id: int,
) -> None:
    """检查候选页 ID 严格递增，并且位于当前游标和固定上界之间。

    校验失败会终止批次且不推进游标，防止重复、倒序或越界 ID 导致漏处理、重复处理
    或分页死循环。
    """

    ids = [candidate.id for candidate in candidates]
    if ids != sorted(set(ids)):
        raise ValueError("Source 候选页必须按唯一 expert id 严格升序排列")
    if ids[0] <= after_id or ids[-1] > upper_bound_id:
        raise ValueError("Source 候选页超出当前全量游标范围")


def _build_batch_error_summary(
    failed_count: int,
    errors: list[dict[str, object]],
) -> dict[str, object] | None:
    """生成不含正文的受限批次错误摘要，供 ``finish_batch`` JSON 字段使用。"""

    if failed_count == 0:
        return None
    return {
        "failed_expert_count": failed_count,
        "sample_errors": errors,
        "sample_truncated": failed_count > len(errors),
    }


def _generate_single_batch_number(expert_id: int) -> str:
    """生成单专家业务批次号，不创建或修改任何专家数据。"""

    timestamp = datetime.now().astimezone().strftime("%Y%m%d%H%M%S%f")
    return f"single_{expert_id}_{timestamp}"


def _generate_full_batch_number() -> str:
    """为未指定批次号的全量任务生成带时间戳的唯一业务编号。"""

    timestamp = datetime.now().astimezone().strftime("%Y%m%d%H%M%S%f")
    return f"full_{timestamp}"
