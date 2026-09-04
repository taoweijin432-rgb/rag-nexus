"""已弃用的领域 Adapter 导入路径。

请改用 ``rag_vector_store.domains``。该兼容转发保留一个发布周期。
"""

from __future__ import annotations

import warnings

warnings.warn(
    "rag_vector_store.core.expert 已弃用，请改用 rag_vector_store.domains",
    DeprecationWarning,
    stacklevel=2,
)

from ..domains import ExpertContractAdapter, ResumeContractAdapter

__all__ = ["ExpertContractAdapter", "ResumeContractAdapter"]
