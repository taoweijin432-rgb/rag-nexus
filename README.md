# RAG Milvus 向量入库模块

这是 RAG 基础建设阶段的一个独立向量入库模块。它只约定 chunk 的数据协议，不依赖具体的切分、解析或 embedding 厂商实现。

整体链路如下：

```text
旧 chunk 生成模块                         专家 Chunk Builder
      │ Iterable[Chunk]                    │ Iterable[PreparedChunk]
      ▼                                    ▼
ChunkIngestionService ── EmbeddingModel ──┐
      │ insert_batch                       │ upsert_batch
      ▼                                    ▼
ChunkRepository ─────── MilvusVectorRepository ── Milvus
```

## 安装

项目使用 `uv` 管理依赖：

```bash
uv sync
```

依赖包括：

- Python 3.10+
- `pydantic` 2.x：定义数据协议和校验规则
- `pymilvus` 2.x：连接并操作 Milvus
- `pytest`：开发测试依赖

## Chunk 接入协议

chunk 生成模块只需要返回 `Chunk`，无需知道 Milvus schema 或 embedding API：

```python
from rag_vector_store import Chunk


def generate_chunks():
    yield Chunk(
        chunk_id="document-001:page-1:chunk-0",
        text="这里是需要检索的文本内容。",
        metadata={
            "document_id": "document-001",
            "page": 1,
            "source": "manual.pdf",
        },
    )
```

协议字段只有三类：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `chunk_id` | `str` | 稳定且唯一的 chunk 主键；后续 upsert/delete 依赖它 |
| `text` | `str` | 需要向量化的文本 |
| `metadata` | `dict[str, Any]` | 文档、页码、来源、租户等业务信息；必须可 JSON 序列化 |

不同解析器或 chunker 的额外信息统一放入 `metadata`，避免把具体 chunk 生成逻辑写进入库模块。

## 接入不同 embedding 模型

实现一个很小的 adapter 即可替换 embedding 模型：

```python
from collections.abc import Sequence


class MyEmbeddingModel:
    dimension = 1536

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        # 调用本地模型或远程 embedding API，并保持输入顺序
        return embedding_client.embed(texts)
```

`dimension` 必须与模型返回的向量长度一致。服务会校验：

- 返回向量数量是否与输入文本数量相同；
- 每个向量是否符合 `dimension`；
- 向量是否包含 NaN 或无穷值。

因此接入新模型时只需替换 adapter，不需要修改 Chunk、service 或 repository。

### 公司云端 Embedding 接入

项目提供 `CompanyCloudEmbeddingModel`，按照公司当前确认的 HTTP JSON 协议调用云端
Embedding 服务。请求使用 `Authorization: Bearer <EMBEDDING_API_KEY>`，请求体为
`model` 和 `input`，响应从 `data[].embedding` 读取。API 地址、密钥和模型配置都从
环境变量读取，真实密钥不要写入代码、README 或 Git。

```bash
export EMBEDDING_API_URL="https://公司实际的接口地址"
export EMBEDDING_API_KEY="在本地填写真实密钥"
export EMBEDDING_MODEL="text-embedding-v4"
export EMBEDDING_MODEL_ALIAS="text-embedding-v4"
export EMBEDDING_DIMENSION="1024"
```

调用入口如下：

```python
from rag_vector_store import CompanyCloudEmbeddingModel


model = CompanyCloudEmbeddingModel.from_env()
embeddings = model.embed(
    [
        "这是第一段测试文本",
        "这是第二段测试文本",
    ]
)

# 配置值来自真实 smoke test，每次 API 响应仍会再次校验维度。
print(model.dimension)
```

适配器会校验返回数量、向量是否为空、向量元素是否为有限数值，以及同一批次的
dimension 是否一致。配置 `EMBEDDING_DIMENSION` 后，ingestion 可以在首次 API 调用前
安全初始化既有 Collection；API 返回维度与配置不一致时会立即报错。未配置时，仍会在
第一次调用成功后从响应获得维度。当前适配器不会自动做 normalize，也不会额外添加
未确认的请求参数。

## 专家 Chunk 接入协议（第一阶段）

本阶段不实现具体的 `ProfileSummaryBuilder`、`EmploymentBuilder` 或
`ProjectBuilder`。同事的 Chunk 模块只需要产出 `Iterable[PreparedChunk]`，并把它
交给 `ChunkIngestionService.ingest_prepared`：

```python
from collections.abc import Iterable

from rag_vector_store import ChunkIngestionService, PreparedChunk


class ColleagueChunkBuilder:
    def build(self, source: object) -> Iterable[PreparedChunk]:
        # 这里由 Chunk 模块负责解析、规范化、分块和生成稳定 ID。
        return build_prepared_chunks(source)


prepared_chunks = ColleagueChunkBuilder().build(source_document)
result = service.ingest_prepared(prepared_chunks)
```

`PreparedChunk` 是向量化边界的数据契约，不包含 `embedding`。它必须携带
`vector_doc_id`、`chunk_id`、`expert_id`、`chunk_type`、`chunk_text`、来源追溯、
hash/version、embedding 模型版本、策略版本、推荐资格和 `metadata_ext` 等字段。
向量化后会生成 `EmbeddedPreparedChunk`，它保留全部原字段，只增加
`embedding: list[float]`。

专家路径的批处理顺序固定为：

```text
Iterable[PreparedChunk]
  → 提取 chunk_text
  → EmbeddingModel.embed
  → 校验数量和向量维度
  → EmbeddedPreparedChunk
  → MilvusVectorRepository.upsert_batch
```

`metadata_ext` 只放低频或未来扩展字段；`expert_id`、`chunk_type`、公司、岗位、
时间、状态和推荐资格等高频字段必须使用 `PreparedChunk` 的显式字段，不能再次复制
到 `metadata_ext`。

## Milvus 配置

复制环境变量模板并按部署环境设置：

```bash
cp .env.example .env
export MILVUS_URI="http://localhost:19530"
export MILVUS_TOKEN="root:Milvus"  # 使用账号密码时不要设置此项
export MILVUS_COLLECTION_NAME="rag_chunks"
export MILVUS_SCHEMA_MODE="legacy"  # 专家集合使用 expert
```

模块不会自动读取 `.env` 文件；生产环境建议由容器编排或进程管理器注入环境变量。支持的连接变量：

- `MILVUS_URI`：Milvus URI，默认 `http://localhost:19530`
- `MILVUS_TOKEN`：`username:password` 格式的 token
- `MILVUS_USER` / `MILVUS_PASSWORD`：账号密码方式，不能与 token 同时设置
- `MILVUS_DB_NAME`：可选数据库名
- `MILVUS_TIMEOUT`：可选请求超时时间（秒）
- `MILVUS_COLLECTION_NAME`：集合名，默认 `rag_chunks`
- `MILVUS_SCHEMA_MODE`：`legacy` 或 `expert`，默认 `legacy`
- `MILVUS_INDEX_TYPE`：向量索引类型，V1.2 DEV 建议使用 `HNSW`
- `MILVUS_METRIC_TYPE`：向量距离，V1.2 DEV 使用 `COSINE`
- `MILVUS_INDEX_PARAMS`：JSON 格式的索引参数，例如 `'{"M":16,"efConstruction":200}'`

## 创建 V1.2 DEV Collection

V1.2 Collection 必须使用独立的 DEV 名称，并将 schema 模式和索引参数显式配置：

```bash
MILVUS_COLLECTION_NAME=expert_candidates_v12_dev
MILVUS_SCHEMA_MODE=expert
MILVUS_INDEX_TYPE=HNSW
MILVUS_METRIC_TYPE=COSINE
MILVUS_INDEX_PARAMS='{"M":16,"efConstruction":200}'
```

先查看不会连接或修改 Milvus 的 Schema 计划：

```bash
uv run --env-file .env python scripts/create_v12_dev_collection.py --dry-run
```

确认后执行创建和只读验收：

```bash
uv run --env-file .env python scripts/create_v12_dev_collection.py
```

脚本只会在目标 Collection 不存在时创建 Collection 和 `embedding` 向量索引；如果
目标已经存在，则只比较 schema/index 并报告差异，绝不会自动 drop、alter 或重建。
脚本不会构造 PreparedChunk，也不会调用 insert、upsert、delete 或写入 Entity。

## 验证 V1.2 DEV 入库与幂等

以下脚本只允许向 `expert_candidates_v12_dev` upsert 三条无 PII 的 DEV fixture，目标名
不匹配、Collection 不存在或 schema/index 不兼容时都会停止，且不会自动创建 Collection：

```bash
uv run --env-file .env python scripts/verify_v12_dev_ingestion.py
```

脚本会真实调用 `CompanyCloudEmbeddingModel`，通过 `ingest_prepared` 执行两次相同 upsert，
按 `vector_doc_id` 回查全部字段，并验证逻辑 Entity 数量、nullable、`metadata_ext` 和幂等性。
它不会打印 API Key、完整向量或 Chunk 正文。

## 写入 Milvus

```python
from rag_vector_store import (
    ChunkIngestionService,
    MilvusSettings,
    MilvusVectorRepository,
)

settings = MilvusSettings.from_env()
embedding_model = MyEmbeddingModel()
repository = MilvusVectorRepository(
    settings,
    vector_dimension=embedding_model.dimension,
)

try:
    service = ChunkIngestionService(
        embedding_model,
        repository,
        batch_size=64,
    )
    result = service.ingest(generate_chunks())
    print(result)
finally:
    repository.close()
```

`ingest` 接收 `Iterable`，所以可以直接消费生成器，不需要一次把所有 chunk 放进内存。每个 batch 的流程是：先批量调用 embedding，再调用一次 repository 的 `insert_batch`。

专家路径请显式调用 `ingest_prepared`，它使用 `upsert_batch`，主键为
`vector_doc_id`。也可以直接调用 `service.ingest(prepared_chunks)`，服务会根据第一条
记录自动分派到专家路径；推荐使用显式方法，让调用意图更清楚。

首次运行时 repository 会创建以下集合字段；如果集合已经存在，则会校验字段和向量维度：

| Milvus 字段 | 类型 | 用途 |
| --- | --- | --- |
| `chunk_id` | `VARCHAR` primary key | 稳定主键 |
| `text` | `VARCHAR` | 原始 chunk 文本 |
| `embedding` | `FLOAT_VECTOR` | 向量 |
| `metadata` | `JSON` | 可扩展元数据 |

默认使用 `AUTOINDEX` 和 `COSINE`。如果有专门的索引需求，可以在构造 `MilvusSettings` 时传入 `index_type`、`metric_type` 和 `index_params`。

当 `MILVUS_SCHEMA_MODE=expert` 时，`MilvusVectorRepository` 使用专家 schema：

| Milvus 字段 | 类型 | 用途 |
| --- | --- | --- |
| `vector_doc_id` | `VARCHAR` primary key | 专家索引主键 |
| `expert_id`、`chunk_id`、`chunk_type` | 标量字段 | 专家和 chunk 标识 |
| `chunk_text` | `VARCHAR` | 向量化前的完整文本 |
| `company_name_norm`、`role_code`、时间和状态字段 | 标量字段 | 高频过滤和排序条件 |
| `source_hash`、`content_hash`、模型/策略版本 | 标量字段 | 追溯和增量判定 |
| `embedding` | `FLOAT_VECTOR` | 向量 |
| `metadata_ext` | `JSON` | 低频、扩展元数据 |

legacy schema（`chunk_id` primary key、`text`、`embedding`、`metadata JSON`）仍然保留。
schema 只由配置或 repository 的显式参数选择；已有集合只做兼容性校验，不会自动
drop、recreate 或 migrate。专家模式应指向已确认的专家 schema 集合；如果目标集合仍是
legacy schema，repository 会报告不兼容，而不会替换它。

## Repository 扩展点

业务层只依赖 `ChunkRepository` 协议，当前 Milvus 实现已经保留并封装了后续写操作：

```python
repository.upsert_batch(embedded_chunks)
repository.delete_by_ids(["document-001:page-1:chunk-0"])
```

专家 schema 下 `upsert_batch` 和 `delete_by_ids` 使用 `vector_doc_id`；legacy schema
下继续使用 `chunk_id`。

如果未来需要接入其他向量数据库，可以实现同样的 `initialize`、`insert_batch`、`upsert_batch` 和 `delete_by_ids`，而不修改 embedding 和 chunk 生成模块。

## 测试和示例

最小测试不需要启动 Milvus，使用内存 repository 和 fake embedding model 验证批处理边界：

```bash
uv run pytest
```

连接真实 Milvus 的最小示例位于 [`examples/minimal_ingest.py`](examples/minimal_ingest.py)：

```bash
uv run python examples/minimal_ingest.py
```

示例中的 `DemoEmbeddingModel` 仅用于演示接口，实际使用时替换成团队选择的 embedding adapter。
