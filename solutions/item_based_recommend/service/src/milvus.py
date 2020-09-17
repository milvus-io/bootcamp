import time
from milvus import *
from src.config import MILVUS_HOST, MILVUS_PORT, TABLE_NAME, collection_param, search_param, top_k


def milvus_client():
	try:
		milvus = Milvus(host=MILVUS_HOST, port=MILVUS_PORT)
		return milvus
	except Exception as e:
		print("Milvus client error:", e)


def has_table(client):
    try:
        status, ok = client.has_collection(collection_name=TABLE_NAME)
        return status, ok
    except Exception as e:
    	print("Milvus has_table error:", e)


def create_table(client):
    try:
        status = client.create_collection(collection_param)
        return status
    except Exception as e:
    	print("Milvus create table error:", e)



def create_index(client):
    param = {'nlist': 16384}
    try:
        status = client.create_index(TABLE_NAME, IndexType.IVF_FLAT, param)
        return status
    except Exception as e:
    	print("Milvus create index error:", e)


def milvus_insert(client, ids_list, vectors):
	try:
		status, ids = client.insert(collection_name=TABLE_NAME, records=vectors, ids=ids_list)
		return status, ids
	except Exception as e:
		print("Milvus insert error:", e)


def milvus_search(client,vec):
	try:
		status, results = client.search(collection_name=TABLE_NAME, query_records=vec, top_k=top_k, params=search_param)
		return status, results
	except Exception as e:
		print("Milvus search error:", e)


def milvus_collection_rows(client):
    try:
        rows = client.count_entities(collection_name=TABLE_NAME)[1]
        return rows
    except Exception as e:
        print("get milvus rows error: ", e)