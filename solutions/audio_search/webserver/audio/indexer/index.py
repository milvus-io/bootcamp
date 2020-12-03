import logging
from milvus import Milvus, DataType
from audio.common.config import MILVUS_HOST, MILVUS_PORT


def milvus_client():
    try:
        milvus = Milvus(host=MILVUS_HOST, port=MILVUS_PORT)
        return milvus
    except Exception as e:
        print("Milvus ERROR:", e)
        logging.error(e)


def create_table_milvus(client, table_name, dimension):
    collection_param = {
    "fields": [
        {"name": "embedding", "type": DataType.FLOAT_VECTOR, "params": {"dim": dimension}},
    ],
    "segment_row_limit": 800000,
    "auto_id": True
    }
    try:
        status = client.create_collection(table_name, collection_param)
        return status
    except Exception as e:
        print("Milvus ERROR:", e)
        logging.error(e)


def insert_vectors(client, table_name, vectors):
    hybrid_entities = [{"name": "embedding", "values": vectors, "type": DataType.FLOAT_VECTOR}]
    try:
        ids = client.insert(table_name, hybrid_entities)
        return ids
    except Exception as e:
        print("Milvus ERROR:", e)
        logging.error(e)


def create_index(client, table_name, metric_type):
    try:
        status = client.create_index(table_name, "embedding",
                    {"index_type": "IVF_FLAT", "metric_type": metric_type, "params": {"nlist": 8192}})
        return status
    except Exception as e:
        print("Milvus ERROR:", e)
        logging.error(e)


def delete_collection(client, table_name):
    try:
        status = client.drop_collection(collection_name=table_name)
        # print(status)
        return status
    except Exception as e:
        print("Milvus ERROR:", e)
        logging.error(e)


def search_vectors(client, table_name, vectors, metric, top_k):
    query_hybrid = {
        "bool": {
            "must": [
                {
                    "vector": {
                        "embedding": {"topk": top_k, "query": vectors, "metric_type": metric}
                    }
                }
            ]
        }
    }
    try:
        res = client.search(table_name, query_hybrid)
        return res
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
        num = client.count_entities(collection_name=table_name)
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