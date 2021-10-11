from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import numpy as np
from multiprocessing import Pool
import time
import logging
import os
import sys
import getopt


MILVUS_HOST = 127.0.0.1   # Milvus service address
MILVUS_PORT = 19530       # Milvus service port

DIM = 128 # Vector dimension
NQ = 10   # The number of the query vectors in each batch
TOP_K = 5 # The number of the vectors that we want to find the nearest neighbors

PROCESS_NUM = 2           # Concurrent number
LOOP = 4                  # The number of query time


#Creat IVF_FLAT index for a collection.
def create_index(collection_name, index_type, index_param):
    pass


# Load a collection data from disk to memory before searching in Milvus.
def load_collection(collection_name):
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
    collection = Collection(name=collection_name)
    collection.load()


# Child process is responsible for query。
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



# Process pool for searching.
def multi_search_pool(collection_name):
	p = Pool(PROCESS_NUM)
    begin_time = time.time()
    for i in range(LOOP):
        p.apply_async(sub_search, (i, collection_name,))
    p.close()
    p.join()
    print("total cost time: {}".format(time.time() - begin_time))
    logging.info("total cost time: {}".format(time.time() - begin_time))



def main():
    try:
        opts, args = getopt.getopt(
            sys.argv[1:],
            "hn:ls",
            ["help", "name=", "load", "search"])
    except getopt.GetoptError:
        print("Error parameters, See 'python main.py --help' for usage.")
        sys.exit(2)

    for opt_name, opt_value in opts:
        if opt_name in ("-h", "--help"):
            print(
                "For parameter descriptions, please refer to "
                "https://github.com/milvus-io/bootcamp/tree/master/benchmark_test/scripts")
            sys.exit(2)
        elif opt_name in ("-n", "--name"):
            collection_name = opt_value
        elif opt_name in ("-l", "--load"):
            load_collection(collection_name)
            sys.exit(2)
        elif opt_name in ("-s", "--search"):
            multi_search_pool(collection_name)
            sys.exit(2)


if __name__ == "__main__":
    main()

