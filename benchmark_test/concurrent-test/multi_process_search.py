from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import numpy as np


MILVUS_HOST = 127.0.0.1
MILVUS_PORT = 19530

DIM = 128
NQ = 10
TOP_K = 5

PROCESS_NUM = 2


def sub_search(task_id, col_name):
	print("task_id {}, sub process {}".format(task_id, os.getpid()))
    vec = np.random.random((NQ, DIM)).tolist()
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
    collection = Collection(name=col_name)
    search_params = {"metric_type": "L2"}
    time_start = time.time()
    results = collection.search(vec,anns_field="embedding", param=search_params, limit=TOP_K)
    time_end = time.time()
    print("task {} cost time: {}".format(task_id, time_end - time_start))
    logging.info("task {}, process {}, search number:{},search time:{}".format(task_id, os.getpid(), NQ, time_end - time_start))



def search():
	pass