import logging as log
from milvus import Milvus, IndexType, MetricType, Status
from common.config import MILVUS_HOST, MILVUS_PORT, VECTOR_DIMENSION


def milvus_client():
    try:
        milvus = Milvus(host=MILVUS_HOST, port=MILVUS_PORT)
        # status = milvus.connect(MILVUS_HOST, MILVUS_PORT)
        return milvus
    except Exception as e:
        log.error(e)


def create_table(client, table_name=None, dimension=VECTOR_DIMENSION,
                 index_file_size=1024, metric_type=MetricType.L2):
    table_param = {
        'collection_name': table_name,
        'dimension': dimension,
        'index_file_size':index_file_size,
        'metric_type': metric_type
    }
    try:
        status = client.create_collection(table_param)
        return status
    except Exception as e:
        log.error(e)


def insert_vectors(client, table_name, vectors):
    if not client.has_collection(collection_name=table_name):
        log.error("collection %s not exist", table_name)
        return
    try:
        status, ids = client.insert(collection_name=table_name, records=vectors)
        return status, ids
    except Exception as e:
        log.error(e)


def create_index(client, table_name):
    param = {'nlist': 16384}
    # status = client.create_index(table_name, param)
    status = client.create_index(table_name, IndexType.IVF_FLAT, param)
    return status


def delete_table(client, table_name):
    status = client.drop_collection(collection_name=table_name)
    print(status)
    return status


def search_vectors(client, table_name, vectors, top_k):
    search_param = {'nprobe': 16}
    status, res = client.search(collection_name=table_name, query_records=vectors, top_k=top_k, params=search_param)
    return status, res


def has_table(client, table_name):
    status = client.has_collection(collection_name=table_name)
    return status


def count_table(client, table_name):
    status, num = client.count_entities(collection_name=table_name)
    return num