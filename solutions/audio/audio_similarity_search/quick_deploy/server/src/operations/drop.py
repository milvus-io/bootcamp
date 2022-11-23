import sys
from logs import LOGGER
from config import DEFAULT_TABLE


def do_drop(table_name, milvus_cli, mysql_cli):
    if not table_name:
        table_name = DEFAULT_TABLE
    try:
        if not milvus_cli.has_collection(table_name):
            return "Collection is not exist"
        status = milvus_cli.delete_collection(table_name)
        mysql_cli.delete_table(table_name)
        return status
    except Exception as e:
        LOGGER.error(f"Error attempting to drop table: {e}")
        sys.exit(1)
