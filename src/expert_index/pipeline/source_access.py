"""人物来源读取和分页游标访问接口。"""

from __future__ import annotations

from typing import cast

from ..models import ExpertAggregate, ExpertCandidate
from ..source import MySQLSourceAdapter
from .helpers import _validate_expert_candidate_page


class _SourceAccessMixin:
    """为构建流程提供人物读取、候选分页和游标解释方法。

    子类可以覆盖这些方法以适配不同数据集。覆盖实现必须保持人物业务 ID、
    来源物理游标和候选页顺序校验三者的语义边界。
    """

    def _get_full_scan_upper_bound(self, max_people: int | None) -> int:
        """读取固定扫描上界；默认使用 Expert Source。

        Args:
            max_people: 可选最大人物数，用于将来源最大 ID 收缩到前 N 个候选。

        Returns:
            本批次允许扫描到的最大物理游标；没有候选时返回 0。
        """

        source = cast(MySQLSourceAdapter, self.source)
        return source.get_full_scan_upper_bound(max_experts=max_people)

    def _load_full_scan_candidates(
        self,
        *,
        after_cursor_id: int,
        upper_bound_cursor_id: int,
        limit: int,
    ) -> list[object]:
        """读取固定上界内的一页候选。

        Args:
            after_cursor_id: 已可靠处理的最后一个物理游标，结果必须严格大于它。
            upper_bound_cursor_id: 批次启动时固定的最大物理游标。
            limit: 本页最多返回的候选数量。

        Returns:
            按物理游标严格升序排列的候选列表。
        """

        source = cast(MySQLSourceAdapter, self.source)
        return list(
            source.load_full_scan_candidates(
                after_id=after_cursor_id,
                upper_bound_id=upper_bound_cursor_id,
                limit=limit,
            )
        )

    def _load_person_candidate(self, person_id: int) -> object:
        """读取一位人物的生命周期候选。

        Args:
            person_id: 下游 Contract 和批次明细使用的人物业务 ID。

        Returns:
            包含人物身份和逻辑删除状态的候选对象。
        """

        source = cast(MySQLSourceAdapter, self.source)
        return source.load_expert_candidate(person_id)

    def _single_person_is_deleted(self, person_id: int) -> bool:
        """判断单人物入口是否直接执行停用分支。

        Args:
            person_id: 待检查的人物业务 ID。

        Returns:
            True 表示跳过正文读取并提交空快照；Expert 默认返回 False。
        """

        del person_id
        return False

    def _load_person_aggregate(self, person_id: int) -> ExpertAggregate:
        """读取一位人物的四表一致性快照。

        Args:
            person_id: 待读取的人物业务 ID。

        Returns:
            人物主档、工作、项目和教育经历的聚合对象。
        """

        source = cast(MySQLSourceAdapter, self.source)
        return source.load_expert(person_id)

    @staticmethod
    def _candidate_person_id(candidate: object) -> int:
        """从候选对象提取状态表及 Contract 使用的人物业务 ID。

        Args:
            candidate: 当前数据集返回的候选对象。

        Returns:
            候选对应的人物业务 ID。
        """

        return cast(ExpertCandidate, candidate).id

    @staticmethod
    def _candidate_cursor_id(candidate: object) -> int:
        """从候选对象提取 Keyset Pagination 使用的物理游标。

        Args:
            candidate: 当前数据集返回的候选对象。

        Returns:
            候选在来源表中的稳定递增游标。
        """

        return cast(ExpertCandidate, candidate).id

    def _validate_candidate_page(
        self,
        candidates: list[object],
        *,
        after_id: int,
        upper_bound_id: int,
    ) -> None:
        """校验 Expert 候选页的顺序和游标边界。

        Args:
            candidates: Source 返回的当前候选页。
            after_id: 读取本页前的可靠游标。
            upper_bound_id: 当前批次固定的扫描上界。

        Raises:
            ValueError: 候选 ID 重复、倒序或超出游标范围。
        """

        _validate_expert_candidate_page(
            [cast(ExpertCandidate, candidate) for candidate in candidates],
            after_id=after_id,
            upper_bound_id=upper_bound_id,
        )
