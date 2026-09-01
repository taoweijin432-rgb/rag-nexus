"""V15 flat JSON Metadata 的领域无关合并工具。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def merge_flat_metadata(
    parts: Sequence[tuple[str, Mapping[str, object]]],
    *,
    physical_keys: frozenset[str] = frozenset(),
    namespace_keys: frozenset[str] = frozenset(),
) -> dict[str, object]:
    """合并多个来源为单层 JSON，重复、物理字段和 namespace 一律拒绝。"""

    merged: dict[str, object] = {}
    origin_of: dict[str, str] = {}
    for origin, mapping in parts:
        for key, value in mapping.items():
            if key in merged:
                raise ValueError(
                    f"flat metadata key 冲突: {key!r} 同时来自 "
                    f"{origin_of[key]} 与 {origin}，禁止覆盖"
                )
            if key in physical_keys:
                raise ValueError(f"一级字段 {key!r} 不得重复出现在 flat metadata JSON")
            if key in namespace_keys:
                raise ValueError(f"flat metadata 禁止 namespace 键 {key!r}")
            merged[key] = value
            origin_of[key] = origin
    return merged


__all__ = ["merge_flat_metadata"]
