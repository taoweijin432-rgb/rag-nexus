# Canal 只读诊断 Client

该独立 Maven 工程用于确认本机 Canal Server 的真实消息协议。它只读取一个 batch，输出
不含字段值的元数据摘要，然后执行 `rollback(batch_id)`。程序不会调用 `ack`，不会写四张
Source 表，也不会推进项目内 CDC Checkpoint。

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
