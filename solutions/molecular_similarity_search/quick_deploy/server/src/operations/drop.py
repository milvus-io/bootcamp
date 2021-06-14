import sys
from src.logs import LOGGER
from src.config import DEFAULT_TABLE


def do_drop(table_name, milvus_cli, mysql_cli):
    if not table_name:
        table_name = DEFAULT_TABLE
    try:
        status = milvus_cli.delete_collection(table_name)
        mysql_cli.delete_table(table_name)
        return status
    except Exception as e:
        LOGGER.error(" Error with  drop table: {}".format(e))
        sys.exit(1)