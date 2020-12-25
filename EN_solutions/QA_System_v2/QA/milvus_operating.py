from milvus import *
import time
import sys
from QA.config import DEFAULT_TABLE as TABLE_NAME
from QA.config import MILVUS_HOST, MILVUS_PORT, collection_param, search_param, top_k


def milvus_client():
    try:
        milvus = Milvus(host=MILVUS_HOST, port=MILVUS_PORT)
        return milvus
    except Exception as e:
        print("Milvus client error:", e)


def has_table(table_name, client):
    try:
        status, ok = client.has_collection(collection_name=table_name)
        print(status)
        return status, ok
    except Exception as e:
        print("Milvus has_table error:", e)


def create_table(table_name, client):
    try:
        collection_param = {
            'collection_name': table_name,
            'dimension': 768,
            'index_file_size':2048,
            'metric_type':  MetricType.IP
        }
        status = client.create_collection(collection_param)
        return status
    except Exception as e:
        print("Milvus create table error:", e)

def drop_milvus_table(table_name, client):
    try:
        status = client.drop_collection(table_name)
    except Exception as e:
        print("Milvus drop table error:", e)


def create_index(table_name, client):
    param = {'nlist': 16384}
    try:
        status = client.create_index(table_name, IndexType.IVF_FLAT, param)
        return status
    except Exception as e:
        print("Milvus create index error:", e)


def milvus_insert(table_name, client, vectors):
    try:
        status, ids = client.insert(collection_name=table_name, records=vectors)
        return status, ids
    except Exception as e:
        print("Milvus insert error:", e)


def milvus_search(client, vec, table_name):
    try:
        status, results = client.search(collection_name=table_name, query_records=vec, top_k=top_k, params=search_param)
        return status, results
    except Exception as e:
        print("Milvus search error:", e)


def get_milvus_rows(client, table_name):
    try:
        status,results = client.count_entities(collection_name=table_name)
        return results
    except Exception as e:
        print("get milvus entity rows error: ", e)
        sys.exit(2)
