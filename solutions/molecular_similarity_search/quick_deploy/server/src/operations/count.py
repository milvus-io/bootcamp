import sys
from src.logs import LOGGER
from src.config import DEFAULT_TABLE


def do_count(table_name, milvus_cli, mysql_cli):
    if not table_name:
        table_name = DEFAULT_TABLE
    try:
        if not milvus_cli.has_collection(table_name):
            return None
        milvus_num = milvus_cli.count(table_name)
        mysql_num = mysql_cli.count_table(table_name)
        LOGGER.debug("The num of Milvus: {} and Mysql: {}".format(milvus_num, mysql_num))
        assert milvus_num == mysql_num
        return milvus_num
    except Exception as e:
        LOGGER.error(" Error with count table {}".format(e))
        sys.exit(1)