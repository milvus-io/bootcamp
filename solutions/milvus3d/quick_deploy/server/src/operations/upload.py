import sys

sys.path.append("..")
from config import TOP_K, DEFAULT_TABLE
from logs import LOGGER


def do_upload(table_name, model_path, model, milvus_client, mysql_cli):
    try:
        if not table_name:
            table_name = DEFAULT_TABLE
        feat = model.do_extract(model_path)
        ids = milvus_client.insert(table_name, [feat])
        milvus_client.create_index(table_name)
        mysql_cli.create_mysql_table(table_name)
        mysql_cli.load_data_to_mysql(table_name, [(str(ids[0]), model_path.encode())])
        return ids[0]
    except Exception as e:
        LOGGER.error(" Error with upload : {}".format(e))
        sys.exit(1)
