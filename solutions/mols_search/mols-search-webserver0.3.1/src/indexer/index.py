import logging as log
from milvus import *
from common.config import MILVUS_HOST, MILVUS_PORT, VECTOR_DIMENSION


def milvus_client():
    try:
        milvus = Milvus()
        status = milvus.connect(MILVUS_HOST, MILVUS_PORT, timeout=60)
        return milvus
    except Exception as e:
        log.error(e)


def create_table(client, table_name=None, dimension=VECTOR_DIMENSION,
                 index_file_size=256, metric_type=MetricType.JACCARD):
    table_param = {
        'table_name': table_name,
        'dimension': dimension,
        'index_file_size':index_file_size,
        'metric_type': metric_type
    }
    try:
        status = client.create_table(table_param)
        return status
    except Exception as e:
        log.error(e)


def insert_vectors(client, table_name, vectors):
    if not client.has_table(table_name=table_name):
        log.error("table %s not exist", table_name)
        return
    try:
        status, ids = client.insert(table_name=table_name, records=vectors)
        return status, ids
    except Exception as e:
        log.error(e)


def create_index(client, table_name):
    param = {'index_type': IndexType.IVFLAT, 'nlist': 2048}
    status = client.create_index(table_name, param)
    return status


def delete_table(client, table_name):
    status = client.delete_table(table_name=table_name)
    print(status)
    return status


def search_vectors(client, table_name, vectors, top_k):
    status, res = client.search_vectors(table_name=table_name, query_records=vectors, top_k=top_k, nprobe=512)
    return status, res


def has_table(client, table_name):
    status = client.has_table(table_name=table_name)
    return status


def count_table(client, table_name):
    status, num = client.get_table_row_count(table_name=table_name)
    print(status)
    # status, num = client.count_table(table_name=table_name)
    return num