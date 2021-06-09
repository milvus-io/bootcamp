import sys

sys.path.append("..")
from config import DEFAULT_TABLE


def do_count(table_name, mil_cli):
    if not table_name:
        table_name = DEFAULT_TABLE
    try:
        num = mil_cli.count(table_name)
        return num
    except Exception as e:
        return "Fail with error {}".format(e)
