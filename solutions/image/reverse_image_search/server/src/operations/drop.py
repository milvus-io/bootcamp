import sys
from config import DEFAULT_TABLE
from logs import LOGGER
from milvus_helpers import MilvusHelper
from mysql_helpers import MySQLHelper


def do_drop(table_name: str, milvus_cli: MilvusHelper, mysql_cli: MySQLHelper):
    if not table_name:
        table_name = DEFAULT_TABLE
    try:
        if not milvus_cli.has_collection(table_name):
            return f"Milvus doesn't have a collection named {table_name}"
        status = milvus_cli.delete_collection(table_name)
        mysql_cli.delete_table(table_name)
        return status
    except Exception as e:
        LOGGER.error(f"Error with drop table: {e}")
        sys.exit(1)
