"""批量编排：MySQL → chunk → JSONL。

推荐以模块方式运行：

    python -m expert_chunk_pipeline.pipeline --max-records 10

不传 ``--max-records`` 表示处理全部未删除专家；开发阶段建议显式限制数量。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .chunk import build_expert_chunk
from .mysql_reader import iter_person_batches

DEFAULT_OUTPUT = Path("data/expert_chunks.jsonl")


def write_chunks(
    output_path: Path,
    batch_size: int = 100,
    after_id: int = 0,
    max_records: int | None = None,
) -> int:
    """读取数据库、构建 chunk 并原子写出 JSONL，返回写入条数。"""

    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")
    if max_records is not None and max_records <= 0:
        raise ValueError("max_records 必须大于 0 或为 None")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    written = 0

    try:
        # 先完整写临时文件，成功后再替换正式文件，避免留下半份结果。
        with temporary_path.open("w", encoding="utf-8") as output_file:
            for persons in iter_person_batches(batch_size, after_id, max_records):
                for person in persons:
                    chunk: dict[str, Any] = build_expert_chunk(person)
                    output_file.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                    written += 1
        temporary_path.replace(output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    return written


def parse_args() -> argparse.Namespace:
    """解析命令行参数；输出路径有默认值，数量限制保持显式可控。"""

    parser = argparse.ArgumentParser(description="从 MySQL 生成专家向量化 chunk")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"输出路径（默认：{DEFAULT_OUTPUT}）",
    )
    parser.add_argument("--batch-size", type=int, default=100, help="每次从数据库读取的条数")
    parser.add_argument("--after-id", type=int, default=0, help="只处理 ID 大于该值的专家")
    parser.add_argument("--max-records", type=int, help="本次最多处理多少条；不传表示全部")
    return parser.parse_args()


def main() -> None:
    """命令行入口。"""

    arguments = parse_args()
    total = write_chunks(
        output_path=arguments.output,
        batch_size=arguments.batch_size,
        after_id=arguments.after_id,
        max_records=arguments.max_records,
    )
    print(f"处理完成，共写入 {total} 条 chunk：{arguments.output}")


if __name__ == "__main__":
    main()
