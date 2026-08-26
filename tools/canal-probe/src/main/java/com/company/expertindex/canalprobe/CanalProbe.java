package com.company.expertindex.canalprobe;

import com.alibaba.otter.canal.client.CanalConnector;
import com.alibaba.otter.canal.client.CanalConnectors;
import com.alibaba.otter.canal.protocol.CanalEntry;
import com.alibaba.otter.canal.protocol.Message;

import java.net.InetSocketAddress;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

/**
 * 只读读取一个 Canal batch，输出协议元数据后回滚 batch。
 *
 * <p>该程序用于确认本机 Canal 的真实消息结构，不执行 ack、不写 Source 表，
 * 也不推进项目内的 CDC Checkpoint。</p>
 */
public final class CanalProbe {
    private CanalProbe() {
    }

    /**
     * 连接本机 Canal，读取一个 batch，输出脱敏摘要并回滚消费位置。
     *
     * @param args 未使用；连接参数通过环境变量提供
     * @throws Exception Canal Client 连接或协议解析失败
     */
    public static void main(String[] args) throws Exception {
        String host = env("CANAL_HOST", "127.0.0.1");
        int port = integerEnv("CANAL_PORT", 11111);
        String destination = env("CANAL_DESTINATION", "company_expert");
        String username = env("CANAL_USERNAME", "");
        String password = env("CANAL_PASSWORD", "");
        String subscription = env(
                "CANAL_SUBSCRIPTION",
                "expert\\.(experts|experts_work_experience|experts_project_experience|experts_education_experience)"
        );
        int batchSize = integerEnv("CANAL_BATCH_SIZE", 10);

        CanalConnector connector = CanalConnectors.newSingleConnector(
                new InetSocketAddress(host, port), destination, username, password
        );
        connector.connect();
        try {
            connector.subscribe(subscription);
            Message message = connector.getWithoutAck(batchSize);
            long batchId = message.getId();
            try {
                printMessage(message, host, port, destination, subscription);
            } finally {
                // 诊断程序永远回滚，确保不会推进 Canal 的确认位点。
                if (batchId != -1L) {
                    connector.rollback(batchId);
                }
            }
        } finally {
            connector.disconnect();
        }
    }

    /** 输出不含行值的 batch 摘要，避免把业务正文或 PII 打到终端。 */
    private static void printMessage(
            Message message,
            String host,
            int port,
            String destination,
            String subscription
    ) throws Exception {
        List<CanalEntry.Entry> entries = message.getEntries();
        System.out.printf(
                "CANAL_BATCH id=%d entries=%d host=%s port=%d destination=%s subscription=%s%n",
                message.getId(), entries.size(), host, port, destination, subscription
        );
        for (int entryIndex = 0; entryIndex < entries.size(); entryIndex++) {
            CanalEntry.Entry entry = entries.get(entryIndex);
            CanalEntry.EntryType entryType = entry.getEntryType();
            CanalEntry.Header header = entry.getHeader();
            System.out.printf(
                    "ENTRY index=%d type=%s schema=%s table=%s logfile=%s offset=%d execute_time=%d server_id=%d%n",
                    entryIndex,
                    entryType,
                    header.getSchemaName(),
                    header.getTableName(),
                    header.getLogfileName(),
                    header.getLogfileOffset(),
                    header.getExecuteTime(),
                    header.getServerId()
            );
            if (entryType == CanalEntry.EntryType.ROWDATA) {
                printRowChange(entry);
            }
        }
    }

    /** 输出行变更类型和字段名，不输出字段值。 */
    private static void printRowChange(CanalEntry.Entry entry) throws Exception {
        CanalEntry.RowChange rowChange = CanalEntry.RowChange.parseFrom(entry.getStoreValue());
        System.out.printf(
                "ROW_CHANGE event=%s rows=%d is_ddl=%s sql_present=%s%n",
                rowChange.getEventType(),
                rowChange.getRowDatasCount(),
                rowChange.getIsDdl(),
                !rowChange.getSql().isEmpty()
        );
        for (int rowIndex = 0; rowIndex < rowChange.getRowDatasCount(); rowIndex++) {
            CanalEntry.RowData rowData = rowChange.getRowDatas(rowIndex);
            System.out.printf(
                    "ROW index=%d before_columns=%s after_columns=%s%n",
                    rowIndex,
                    columnNames(rowData.getBeforeColumnsList()),
                    columnNames(rowData.getAfterColumnsList())
            );
        }
    }

    /** 返回稳定排序的字段名列表，禁止携带列值。 */
    private static String columnNames(List<CanalEntry.Column> columns) {
        return columns.stream()
                .map(CanalEntry.Column::getName)
                .filter(Objects::nonNull)
                .collect(Collectors.joining(",", "[", "]"));
    }

    /** 读取非空环境变量，否则使用明确的诊断默认值。 */
    private static String env(String name, String defaultValue) {
        String value = System.getenv(name);
        return value == null || value.isBlank() ? defaultValue : value;
    }

    /** 读取正整数环境变量并拒绝非法值。 */
    private static int integerEnv(String name, int defaultValue) {
        String value = System.getenv(name);
        if (value == null || value.isBlank()) {
            return defaultValue;
        }
        try {
            int parsed = Integer.parseInt(value);
            if (parsed < 1) {
                throw new IllegalArgumentException(name + " 必须是正整数");
            }
            return parsed;
        } catch (NumberFormatException error) {
            throw new IllegalArgumentException(name + " 必须是整数", error);
        }
    }
}
