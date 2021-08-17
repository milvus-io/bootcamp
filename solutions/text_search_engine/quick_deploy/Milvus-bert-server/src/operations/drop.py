import sys

sys.path.append("..")
from config import DEFAULT_TABLE
from logs import LOGGER


def do_drop(table_name, milvus_cli, mysql_cli):
    if not table_name:
        table_name = DEFAULT_TABLE
    try:
        if not milvus_cli.has_collection(table_name):
            return None

        status = milvus_cli.delete_collection(table_name)
        mysql_cli.delete_table(table_name)
        return status
    except Exception as e:
        LOGGER.error(" Error with  drop table: {}".format(e))
        sys.exit(1)
