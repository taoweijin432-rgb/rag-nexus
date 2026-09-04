"""Resume Contract 到通用向量内核的领域适配器。

Resume 当前仍复用 Expert Contract 的内存结构，但持久化身份固定为
``resume_id`` / ``resume:``。领域差异只存在于本 Adapter，不进入通用核心。
"""

from __future__ import annotations

from typing import Any

from .expert import ExpertContractAdapter


class ResumeContractAdapter(ExpertContractAdapter):
    """把兼容 Contract 映射到 Resume 的独立身份空间。"""

    def __init__(
        self,
        validator: Any | None = None,
        eligibility_validator: Any | None = None,
    ) -> None:
        super().__init__(
            validator,
            eligibility_validator,
            storage_owner_field="resume_id",
            id_prefix="resume:",
        )

__all__ = ["ResumeContractAdapter"]
