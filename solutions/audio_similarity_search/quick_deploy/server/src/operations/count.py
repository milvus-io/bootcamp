import sys
from src.logs import LOGGER
from src.config import DEFAULT_TABLE


def do_count(table_name, milvus_cli):
    if not table_name:
        table_name = DEFAULT_TABLE
    try:
        num = milvus_cli.count(table_name)
        return num
    except Exception as e:
        LOGGER.error(" Error attempting to count table {}".format(e))
        sys.exit(1)
