import logging
from milvus import Milvus, IndexType, MetricType, Status
from common.config import MILVUS_HOST, MILVUS_PORT, VECTOR_DIMENSION, TOP_K


def milvus_client():
    try:
        milvus = Milvus(host=MILVUS_HOST, port=MILVUS_PORT)
        # status = milvus.connect(MILVUS_HOST, MILVUS_PORT)
        return milvus
    except Exception as e:
        print("Milvus ERROR:", e)
        logging.error(e)


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
        print("Milvus ERROR:", e)
        logging.error(e)


def insert_vectors(client, table_name, vectors, ids):
    if not client.has_collection(collection_name=table_name):
        logging.error("collection %s not exist", table_name)
        return
    try:
        status, ids = client.insert(collection_name=table_name, records=vectors, ids=ids)
        return status, ids
    except Exception as e:
        print("Milvus ERROR:", e)
        logging.error(e)


def create_index(client, table_name):
    try:
        param = {'nlist': 16384}
        # status = client.create_index(table_name, param)
        status = client.create_index(table_name, IndexType.IVF_FLAT, param)
        return status
    except Exception as e:
        print("Milvus ERROR:", e)
        logging.error(e)


def delete_collection(client, table_name):
    try:
        status = client.drop_collection(collection_name=table_name)
        print(status)
        return status
    except Exception as e:
        print("Milvus ERROR:", e)
        logging.error(e)


def search_vectors(client, table_name, vectors, top_k=TOP_K):
    try:
        search_param = {'nprobe': 16}
        status, res = client.search(collection_name=table_name, query_records=vectors, top_k=top_k, params=search_param)
        return status, res
    except Exception as e:
        print("Milvus ERROR:", e)
        logging.error(e)


def has_table(client, table_name):
    try:
        status = client.has_collection(collection_name=table_name)
        return status
    except Exception as e:
        print("Milvus ERROR:", e)
        logging.error(e)


def count_collection(client, table_name):
    try:
        status, num = client.count_entities(collection_name=table_name)
        return num
    except Exception as e:
        print("Milvus ERROR:", e)
        logging.error(e)


def delete_vectors(client, table_name, ids):
    try:
        status = client.delete_entity_by_id(table_name, ids)
        return status
    except Exception as e:
        print("Milvus ERROR:", e)
        logging.error(e)


def get_vector_by_ids(client, table_name, ids):
    try:
        status, vector = client.get_entity_by_id(collection_name=table_name, ids=ids)
        return status, vector
    except Exception as e:
        print("Milvus ERROR:", e)
        logging.error(e)