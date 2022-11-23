import sys
sys.path.append("..")
from logs import LOGGER
from config import DEFAULT_TABLE


def do_count(table_name, milvus_cli, mysql_cli):
    if not table_name:
        table_name = DEFAULT_TABLE
    try:
        if not milvus_cli.has_collection(table_name):
            return None
        milvus_num = milvus_cli.count(table_name)
        mysql_num = mysql_cli.count_table(table_name)
        LOGGER.debug(f"The num of Milvus: {milvus_num} and Mysql: {mysql_num}")
        return milvus_num
    except Exception as e:
        LOGGER.error(f" Error with count table {e}")
        sys.exit(1)
