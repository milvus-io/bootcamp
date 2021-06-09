import sys

sys.path.append("..")
from config import DEFAULT_TABLE


def do_drop(table_name, mil_cli, mysql_cli):
    if not table_name:
        table_name = DEFAULT_TABLE
    try:
        status = mil_cli.delete_collection(table_name)
        mysql_cli.delete_table(table_name)
        return status
    except Exception as e:
        return "Error with {}".format(e)
