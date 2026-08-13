"""单专家 Chunk 加工流水线的总编排模块。

本模块按照以下固定顺序组织一次完整的专家数据加工：

``Source → Normalize/Stage → Chunk → Validator → MySQL``

各步骤的具体职责由其他模块承担：

* ``source.py``：从 ``experts`` 及工作、项目、教育三张经历表读取同一位专家；
* ``normalize.py``：清洗源字段，并将源数据转换为结构稳定的 Stage 中间对象；
* ``chunk_builder.py``：按照 profile_summary、employment、project 三类规则生成
  Chunk Text、Metadata、来源追溯信息和版本信息；
* ``validators.py``：在入库前执行数据契约、长度、哈希、ID 和 PII 等校验；
* ``repository.py``：管理批次记录，并通过数据库事务保存 Stage 与最终 Chunk。

因此，``pipeline.py`` 负责的是“先做什么、后做什么、失败时如何收口”，不重复
实现 SQL、字段清洗或文本拼接规则。出现异常时，批次会被标记为 ``failed``；
Stage 和 Chunk 的写入事务则由 Repository 负责回滚，避免产生只写入一半的数据。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .chunk_builder import ChunkFactory
from .config import Settings
from .normalize import normalize_aggregate
from .repository import MySQLIndexRepository
from .source import MySQLSourceAdapter
from .validators import validate_chunks


@dataclass(frozen=True)
class BuildResult:
    """一次单专家加工任务的安全结果摘要。

    该对象由 :meth:`ExpertIndexPipeline.build_one` 返回。命令行入口 ``cli.py``
    会使用 ``asdict(result)`` 将其转换为字典，再以 JSON 形式打印到终端。

    这里不保存 Chunk 正文、工作描述或项目描述，只包含批次标识和数量统计，
    因而适合打印到控制台、任务日志，或作为未来 HTTP API 的响应内容。

    ``frozen=True`` 表示对象创建后字段不可修改，避免任务完成后的统计结果被调用方
    意外覆盖。它不是数据库 ORM 模型，也不会自动写入数据库；对应的批次统计由
    ``repository.finish_batch()`` 显式保存。
    """

    batch_id: int  # 批次表自增主键，由 start_batch() 插入批次后返回，用于数据库回查。
    batch_no: str  # 可读的业务批次号；可由调用方传入，也可由 _batch_no() 自动生成。
    expert_id: int  # 本次处理的专家 ID，对应源表 experts.id 和经历表中的 expert_id。
    extracted_count: int  # 记录总数：1 条主档 + 工作 + 项目 + 教育经历数。
    stage_count: int  # 成功保存的 Stage 中间记录数量，由 persist_expert() 返回。
    chunk_count: int  # 本次生成并准备落库的最终 Chunk 数量；长文本拆分时可大于 Stage 数。
    partial_chunk_count: int  # 兼容旧 CLI 结果；V1.3 不再使用 partial，当前恒为 0。


class ExpertIndexPipeline:
    """单专家加工流程的总调度器。

    默认情况下，本类会根据 ``settings`` 创建真实的 MySQL Source Adapter 和
    Repository。构造函数也允许外部传入替代实现，这种做法称为“依赖注入”：

    * 生产运行时不传，使用真实 MySQL；
    * 单元测试时可以传入 Fake Source/Repository，隔离数据库并精确模拟异常；
    * 将来更换数据来源或存储实现时，不需要修改 ``build_one()`` 的编排顺序。
    """

    def __init__(
        self,
        settings: Settings,
        *,
        source: MySQLSourceAdapter | None = None,
        repository: MySQLIndexRepository | None = None,
    ) -> None:
        # 保存本次流水线使用的数据库连接、时区和版本配置。ChunkFactory 后续还会用到
        # chunk_policy_version、metadata_schema_version、dictionary_version 等版本号。
        self.settings = settings

        # ``source or ...`` 的含义：调用方传入 Source 时优先使用传入对象；没有传入时，
        # 才根据 source_db 创建真实 MySQL 读取器。Source 只负责读取四张业务源表。
        self.source = source or MySQLSourceAdapter(settings.source_db)

        # Repository 负责批次表、Stage 表和 expert_chunk 表。source_db 与 index_db
        # 当前可以指向同一测试库，正式环境中也可以分别指向业务库和索引加工库。
        self.repository = repository or MySQLIndexRepository(settings.index_db)

        # ChunkFactory 只接收已经标准化的 Stage 对象，根据三类模板生成最终 Chunk；
        # 它不会直接查询数据库，也不会自行执行入库操作。
        self.factory = ChunkFactory(settings)

    def build_one(self, expert_id: int, *, data_version: str, batch_no: str | None = None) -> BuildResult:
        """完整重建一位专家，并返回本次运行的统计摘要。

        Parameters
        ----------
        expert_id:
            要处理的专家主键，对应 ``experts.id``。Source Adapter 会使用同一个值查询
            ``experts_work_experience.expert_id`` 和
            ``experts_project_experience.expert_id``，从而关联三张源表。
        data_version:
            本次所使用业务数据的版本标签，例如 ``20260812_VECTOR_DOC_ID_V1``。
            它会进入 Stage 和 Chunk 的版本信息，用于重建、审计和问题追踪；它不是
            MySQL 自动生成的值，因此调用方必须明确提供非空字符串。
        batch_no:
            可选的任务业务编号。调用方不传时，由 ``_batch_no(expert_id)`` 根据专家 ID
            与当前时间自动生成。批次号只是任务追踪信息，不会创建任何虚假专家数据。

        Returns
        -------
        BuildResult
            只包含批次 ID、专家 ID 和各阶段数量。CLI 会将其打印为 JSON；真正的
            Chunk Text + Metadata 则由 Repository 保存到 ``expert_chunk`` 表。

        Notes
        -----
        同一来源会产生稳定的 ``chunk_id``。重复运行同一专家时，Repository 使用
        Upsert 更新对应记录，而不是无条件插入重复 Chunk。
        """

        # strip() 可以同时拦截空字符串和只包含空格的字符串。没有 data_version 时，
        # 后续无法判断 Stage/Chunk 属于哪一批业务数据，所以任务在建批次前直接拒绝。
        if not data_version.strip():
            raise ValueError("data_version 不能为空")

        # 调用方提供 batch_no 时原样使用；未提供或值为空时，才自动生成唯一性较高的
        # 批次号。这里生成的是运行标识，不是源业务数据或测试数据。
        actual_batch_no = batch_no or _batch_no(expert_id)

        # 在执行源数据读取前先创建 running 批次。即使后续读取、清洗或入库失败，
        # 数据库中仍能留下本次任务的开始记录和最终失败摘要。
        # 第二个参数 ``single`` 表示当前是“单专家重建”运行模式。
        batch_id = self.repository.start_batch(actual_batch_no, "single")

        # 计数器先初始化为 0。若中途失败，except 分支可以将任务实际推进到的数量
        # 写回批次表，帮助判断失败发生在读取、标准化、生成还是持久化阶段。
        extracted_count = stage_count = chunk_count = 0
        try:
            # 1. Source：按照 expert_id 查询专家主档、工作、项目和教育经历，并聚合为
            # ExpertAggregate。若 experts 主档不存在，Source Adapter 会直接抛出异常。
            aggregate = self.source.load_expert(expert_id)

            # 源记录数量包含：1 条专家主档 + 工作 + 项目 + 教育经历。
            extracted_count = (
                1
                + len(aggregate.employments)
                + len(aggregate.projects)
                + len(aggregate.educations)
            )

            # 2. Normalize/Stage：清理文本、标准化公司/岗位/日期、生成来源哈希，并将
            # 三类 Source 对象转换为 ProfileStage、EmploymentStage、ProjectStage。
            # data_version 和 build_batch_id 会随 Stage 保存，形成数据版本与批次追踪。
            profile, employments, projects = normalize_aggregate(
                aggregate,
                data_version=data_version,
                build_batch_id=batch_id,
            )

            # 3. Chunk Builder：根据三类 Stage 生成 profile_summary、employment、
            # project Chunk。一个过长 Stage 可能被拆成多个 Chunk，因此 chunk_count
            # 并不保证永远等于 stage_count。
            chunks = self.factory.build_all(profile, employments, projects)

            # “没有异常”不等于“成功产出”。空 Chunk 集合无法供后续 Embedding 使用，
            # 因此把它明确视为任务失败，防止产生无结果但状态为 success 的批次。
            if not chunks:
                raise ValueError(f"专家 {expert_id} 没有可生成的 Chunk")

            # 4. Validator：在写入数据库之前，集中检查 Chunk 文本长度、字符数、哈希、
            # Metadata 必填字段、vector_doc_id 公式、来源追溯字段、分片连续性和 PII。
            # 任何一条不符合契约都会抛出异常，因此非法 Chunk 不会进入持久化步骤。
            validate_chunks(chunks)

            # 5. Repository：在同一个数据库事务中保存一位专家的 Stage 和 Chunk。
            # 任一步 SQL 失败时，persist_expert() 负责回滚，避免只写入部分派生数据。
            # 返回值是成功处理的 Stage 数量，而不是 Chunk 数量。
            stage_count = self.repository.persist_expert(profile, employments, projects, chunks)

            # Chunk 已通过校验并完成事务保存，因此此时才记录最终 Chunk 数量。
            chunk_count = len(chunks)

            # V1.3 固定使用 source_summary（画像）和 verified（任职/项目），不再使用
            # parse_status/partial 推断枚举。为保持 CLI 返回结构兼容，该计数暂时固定为 0。
            partial_count = 0

            # V1.3 当前不会生成 partial，保留条件分支仅为了兼容旧结果结构。
            self.repository.finish_batch(
                batch_id,
                status="partial" if partial_count else "success",
                extracted_count=extracted_count,
                stage_count=stage_count,
                chunk_count=chunk_count,
                rejected_count=partial_count,
                error_summary={
                    "note": "兼容分支：发现旧规则 partial Chunk"
                } if partial_count else None,
            )

            # 返回给调用方的是安全摘要，不是 Chunk 列表。CLI 会把这个 dataclass 转成
            # JSON 打印；最终 Chunk 数据仍以 expert_chunk 表为权威输出。
            return BuildResult(
                batch_id=batch_id,
                batch_no=actual_batch_no,
                expert_id=expert_id,
                extracted_count=extracted_count,
                stage_count=stage_count,
                chunk_count=chunk_count,
                partial_chunk_count=partial_count,
            )
        except Exception as exc:
            # 任意环节出现异常时，把已经统计出的阶段数量写回批次表，并将状态标记为
            # failed。注意：Stage/Chunk 的事务回滚由 persist_expert() 自己负责；这里
            # 负责的是批次状态收口，两者属于不同职责。
            #
            # 错误摘要只记录异常类型和最多 400 个字符的异常消息，不主动保存专家背景、
            # 工作描述、项目描述或 Chunk 正文。校验异常也被设计为不携带完整正文，
            # 降低真实专家资料通过任务日志泄露的风险。
            self.repository.finish_batch(
                batch_id,
                status="failed",
                extracted_count=extracted_count,
                stage_count=stage_count,
                chunk_count=chunk_count,
                rejected_count=1,
                error_summary={"error_type": type(exc).__name__, "message": str(exc)[:400]},
            )
            # 记录批次失败后继续抛出原异常，让 CLI、调度平台或 API 调用方能够收到失败，
            # 而不是把已经失败的任务伪装成正常返回。
            raise


def _batch_no(expert_id: int) -> str:
    """生成单专家任务的默认业务批次号。

    批次号格式为 ``single_{expert_id}_{年月日时分秒微秒}``，例如：
    ``single_999641217_20260812153045123456``。

    它表示“系统在某个时间执行了一次专家加工任务”，只用于运行追踪、重放和排障；
    不会创建或修改 ``experts``、工作经历、项目经历等业务数据。微秒部分用于降低同一
    专家在短时间内连续运行时发生批次号重复的概率。
    """

    # astimezone() 使用运行环境当前时区；项目默认环境是 Asia/Shanghai。
    # %f 表示六位微秒，使两次紧邻执行通常也能得到不同批次号。
    timestamp = datetime.now().astimezone().strftime("%Y%m%d%H%M%S%f")
    return f"single_{expert_id}_{timestamp}"
