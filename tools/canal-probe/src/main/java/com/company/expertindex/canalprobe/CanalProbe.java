package com.company.expertindex.canalprobe;

import com.alibaba.otter.canal.client.CanalConnector;
import com.alibaba.otter.canal.client.CanalConnectors;
import com.alibaba.otter.canal.protocol.CanalEntry;
import com.alibaba.otter.canal.protocol.Message;

import java.net.InetSocketAddress;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.stream.Collectors;

/**
 * 读取一个 Canal batch，输出协议元数据；默认回滚，显式 ACK 模式仅用于受控联调。
 *
 * <p>程序不写 Source 表，也不推进项目内的 CDC Checkpoint。ACK 模式要求同时提供
 * 精确 batch、表、事件、行数和确认短语，避免误确认未知消息。</p>
 */
public final class CanalProbe {
    private CanalProbe() {
    }

    /**
     * 连接 Canal，读取一个 batch，输出脱敏摘要；默认回滚，ACK 模式在校验通过后确认。
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
        String action = env("CANAL_PROBE_ACTION", "rollback").toLowerCase(Locale.ROOT);
        if (!action.equals("rollback") && !action.equals("ack")) {
            throw new IllegalArgumentException("CANAL_PROBE_ACTION 只能是 rollback 或 ack");
        }

        CanalConnector connector = CanalConnectors.newSingleConnector(
                new InetSocketAddress(host, port), destination, username, password
        );
        connector.connect();
        try {
            connector.subscribe(subscription);
            Message message = connector.getWithoutAck(batchSize);
            long batchId = message.getId();
            boolean ackAttempted = false;
            try {
                printMessage(message, host, port, destination, subscription);
                if (action.equals("ack")) {
                    validateAckRequest(message, batchId);
                    // ACK 调用返回前结果可能已到达 Canal；异常时禁止再 rollback，避免覆盖确认结果。
                    ackAttempted = true;
                    connector.ack(batchId);
                    System.out.printf("CANAL_ACK status=success batch_id=%d%n", batchId);
                }
            } finally {
                // 默认回滚；校验失败也回滚；ACK 发起后交由 Canal 保持最终状态。
                if (batchId != -1L && !ackAttempted) {
                    connector.rollback(batchId);
                }
            }
        } finally {
            connector.disconnect();
        }
    }

    /** 校验 ACK 所需的精确目标，拒绝空 batch、混合表或未声明的事件。 */
    private static void validateAckRequest(Message message, long batchId) throws Exception {
        long expectedBatchId = longEnv("CANAL_PROBE_EXPECTED_BATCH_ID", -1L);
        String expectedTable = requiredEnv("CANAL_PROBE_EXPECTED_TABLE");
        List<String> expectedEvents = expectedEvents();
        int expectedRows = integerEnv("CANAL_PROBE_EXPECTED_ROWS", -1);
        String confirmation = requiredEnv("CANAL_PROBE_ACK_CONFIRM");
        if (!confirmation.equals("I_UNDERSTAND_ACK_TEST")) {
            throw new IllegalArgumentException("CANAL_PROBE_ACK_CONFIRM 不正确，拒绝 ACK");
        }
        if (batchId < 0L || batchId != expectedBatchId) {
            throw new IllegalStateException(
                    "batch_id 不匹配，实际=" + batchId + " 期望=" + expectedBatchId
            );
        }

        int rowDataEntries = 0;
        int totalRows = 0;
        for (CanalEntry.Entry entry : message.getEntries()) {
            if (entry.getEntryType() != CanalEntry.EntryType.ROWDATA) {
                continue;
            }
            rowDataEntries++;
            if (!expectedTable.equals(entry.getHeader().getTableName())) {
                throw new IllegalStateException("ROWDATA 表不匹配，实际=" + entry.getHeader().getTableName());
            }
            CanalEntry.RowChange rowChange = CanalEntry.RowChange.parseFrom(entry.getStoreValue());
            if (rowChange.getIsDdl() || rowDataEntries > expectedEvents.size()
                    || !expectedEvents.get(rowDataEntries - 1).equals(rowChange.getEventType().name())) {
                throw new IllegalStateException("ROWDATA 事件不匹配，实际=" + rowChange.getEventType());
            }
            totalRows += rowChange.getRowDatasCount();
        }
        if (rowDataEntries == 0 || rowDataEntries != expectedEvents.size() || totalRows != expectedRows) {
            throw new IllegalStateException(
                    "ROWDATA 结构不匹配，事件数=" + rowDataEntries + " 期望=" + expectedEvents.size()
                            + "，行数=" + totalRows + " 期望=" + expectedRows
            );
        }
    }

    /** 读取按 ROWDATA 顺序声明的事件序列；单事件变量保持向后兼容。 */
    private static List<String> expectedEvents() {
        String sequence = System.getenv("CANAL_PROBE_EXPECTED_EVENTS");
        if (sequence != null && !sequence.isBlank()) {
            List<String> events = Arrays.stream(sequence.split(",", -1))
                    .map(String::trim)
                    .filter(value -> !value.isEmpty())
                    .map(value -> value.toUpperCase(Locale.ROOT))
                    .collect(Collectors.toList());
            if (events.isEmpty()) {
                throw new IllegalArgumentException("CANAL_PROBE_EXPECTED_EVENTS 不能为空");
            }
            return events;
        }
        return List.of(requiredEnv("CANAL_PROBE_EXPECTED_EVENT").toUpperCase(Locale.ROOT));
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

    /** 返回按 Canal 原始顺序排列的字段名列表，禁止携带列值。 */
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

    /** 读取 ACK 必填环境变量，缺失时立即失败。 */
    private static String requiredEnv(String name) {
        String value = System.getenv(name);
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(name + " 是 ACK 模式必填项");
        }
        return value;
    }

    /** 读取长整型环境变量并拒绝非法值。 */
    private static long longEnv(String name, long defaultValue) {
        String value = System.getenv(name);
        if (value == null || value.isBlank()) {
            return defaultValue;
        }
        try {
            return Long.parseLong(value);
        } catch (NumberFormatException error) {
            throw new IllegalArgumentException(name + " 必须是整数", error);
        }
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
