"""具体业务领域到通用向量内核的 Adapter。"""

from .expert import ExpertContractAdapter
from .resume import ResumeContractAdapter

__all__ = ["ExpertContractAdapter", "ResumeContractAdapter"]
