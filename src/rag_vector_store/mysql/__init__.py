"""测试和通用只读场景使用的 MySQL 连接基础设施。"""

from .connection import MySQLConnection, MySQLConnectionError

__all__ = [
    "MySQLConnection",
    "MySQLConnectionError",
]
