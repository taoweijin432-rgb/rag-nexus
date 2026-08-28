# Canal 诊断 Client

该独立 Maven 工程用于确认 Canal Server 的真实消息协议。默认只读取一个 batch，输出
不含字段值的元数据摘要，然后执行 `rollback(batch_id)`。程序不会写四张 Source 表，也
不会推进项目内 CDC Checkpoint。

ACK 是显式的受控联调模式，不是默认行为。只有 batch id、表名、事件类型、行数和固定
确认短语全部匹配时才会调用 `ack(batch_id)`；ACK 调用发生异常时不再自动 rollback，避免
覆盖 Canal 端可能已经生效的确认结果。

## 使用条件

- 本机 Canal Server 已启动并监听 `127.0.0.1:11111`；
- destination 默认是 `company_expert`；
- Java 11+ 和 Maven；
- 当前 Canal Server 为 1.1.8，Client 依赖保持同版本。

## 运行

```bash
cd tools/canal-probe
mvn -q compile
mvn -q dependency:copy-dependencies -DoutputDirectory=target/dependency
java -cp "target/classes:target/dependency/*" \
  com.company.expertindex.canalprobe.CanalProbe
```

可通过环境变量覆盖连接参数：

```text
CANAL_HOST=127.0.0.1
CANAL_PORT=11111
CANAL_DESTINATION=company_expert
CANAL_USERNAME=
CANAL_PASSWORD=
CANAL_SUBSCRIPTION=expert\\.(experts|experts_work_experience|experts_project_experience|experts_education_experience)
CANAL_BATCH_SIZE=10
```

输出中的 `logfile/offset/server_id` 只用于确认真实 Offset 组成；`before_columns` 和
`after_columns` 只输出字段名，不输出字段值。若 batch 为空，程序仍会安全退出并断开连接。

## 受保护 ACK 联调

先用默认模式读取并回滚，确认输出中的表、事件和行数后，再将该次输出的 `batch_id + 1`
作为下一次读取的期望值执行 ACK（Canal 每次 `getWithoutAck` 都会分配新的连接内 batch id）：

```bash
export CANAL_PROBE_ACTION=ack
export CANAL_PROBE_EXPECTED_BATCH_ID=<上一次输出的 batch_id 加 1>
export CANAL_PROBE_EXPECTED_CHANGES=experts_education_experience:INSERT:1,experts_education_experience:DELETE:1
export CANAL_PROBE_EXPECTED_ROWS=2
export CANAL_PROBE_ACK_CONFIRM=I_UNDERSTAND_ACK_TEST
java -cp "target/classes:target/dependency/*" \
  com.company.expertindex.canalprobe.CanalProbe
```

ACK 成功后输出 `CANAL_ACK status=success`。完成联调后清理上述 ACK 环境变量，恢复默认
回滚模式。

单一 ROWDATA 也可使用兼容变量 `CANAL_PROBE_EXPECTED_TABLE`、`CANAL_PROBE_EXPECTED_EVENT`
和 `CANAL_PROBE_EXPECTED_ROWS`；多个 ROWDATA 应使用 `CANAL_PROBE_EXPECTED_CHANGES` 完整
声明 `table:event:rows` 顺序，不能只声明其中一条。设置完整清单后不再需要
`CANAL_PROBE_EXPECTED_TABLE` 或 `CANAL_PROBE_EXPECTED_EVENT`。
