import logging
import time
from common.config import DEFAULT_TABLE
from common.const import default_cache_dir

from encoder.encode import feature_extract
from diskcache import Cache
from indexer.index import milvus_client, create_table, insert_vectors, delete_table, search_vectors, create_index,has_table


def do_load(table_name, database_path):
    if not table_name:
        table_name = DEFAULT_TABLE
    cache = Cache(default_cache_dir)
    try:
        vectors, names = feature_extract(table_name, database_path)
        print("start connetc to milvus")
        index_client = milvus_client()
        status, ok = has_table(index_client, table_name)
        if not ok:
            print("create table.")
            create_table(index_client, table_name=table_name)
        print("insert into:", table_name)

        # status, ids = insert_vectors(index_client, table_name, vectors)
        total_ids = []
        ids_lens = 0
        while ids_lens<len(vectors) :
            try:
                status, ids = insert_vectors(index_client, table_name, vectors[ids_lens:ids_lens+100000])
            except:
                status, ids = insert_vectors(index_client, table_name, vectors[ids_lens:len(vectors)])
            ids_lens += 100000
            total_ids += ids
            print("ids:",len(ids))

        create_index(index_client, table_name)
        for i in range(len(names)):
            cache[total_ids[i]] = names[i]
        print("FP finished")
        return "FP finished"
    except Exception as e:
        logging.error(e)
        return "Error with {}".format(e)
    finally:
        if index_client:
            index_client.close()
