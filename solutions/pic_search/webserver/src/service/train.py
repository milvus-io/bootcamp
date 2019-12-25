import logging
import time
from common.config import DEFAULT_TABLE
from common.const import default_cache_dir
# from common.config import DATA_PATH as database_path
from encoder.encode import feature_extract
from preprocessor.vggnet import VGGNet
from diskcache import Cache
from indexer.index import milvus_client, create_table, insert_vectors, delete_table, search_vectors, create_index,has_table


def do_train(table_name, database_path):
    if not table_name:
        table_name = DEFAULT_TABLE
    cache = Cache(default_cache_dir)
    try:
        vectors, names = feature_extract(database_path, VGGNet())
        print("start connetc to milvus")
        index_client = milvus_client()
        # delete_table(index_client, table_name=table_name)
        # time.sleep(1)
        status, ok = has_table(index_client, table_name)
        if not ok:
            print("create table.")
            create_table(index_client, table_name=table_name)
        print("insert into:", table_name)
        status, ids = insert_vectors(index_client, table_name, vectors)
        create_index(index_client, table_name)
        for i in range(len(names)):
            # cache[names[i]] = ids[i]
            cache[ids[i]] = names[i]
        print("Train finished")
        return "Train finished"
    except Exception as e:
        logging.error(e)
        return "Error with {}".format(e)


