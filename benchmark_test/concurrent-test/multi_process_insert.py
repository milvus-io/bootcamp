from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import numpy as np
from multiprocessing import Process
from multiprocessing import Pool, cpu_count
import time
import logging
import os
import sys

logging.basicConfig(filename='benchmark.log', level=logging.DEBUG)

MILVUS_HOST = "127.0.0.1" # Milvus service address
MILVUS_PORT = 19530       # Milvus service port
DIM = 768 # Vector dimension

SHARD_NUM = 5 # Parameter for creating a Milvus collection, Corresponds to how many active datanodes can be used on insert 

TOTAL_NUM = 1000000 # Total number of inserted vectors
BATCH_SIZE = 10000  # The number of vectors inserted in each batch
PROCESS_NUM = 10    # Concurrent number

def create_collection(collection_name, dim=768):
    try:
        connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
        if not utility.has_collection(collection_name):
            field1 = FieldSchema(name="id", dtype=DataType.INT64, descrition="int64", is_primary=True, auto_id=True)
            field2 = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, descrition="float vector", dim=dim,
                                 is_primary=False)
            schema = CollectionSchema(fields=[field1, field2], description="collection description")
            collection = Collection(name=collection_name, schema=schema, shards_num=SHARD_NUM)
            print("Create Milvus collection: {}".format(collection))
            return collection
    except Exception as e:
        logging.error("Failed to create milvus collection: {}".format(e))
        sys.exit(1)


def sub_insert(task_id, col_name):
    print("task_id {}, sub process {}".format(task_id, os.getpid()))
    vec = np.random.random((BATCH_SIZE, DIM)).tolist()
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
    collection = Collection(name=col_name)
    time_start = time.time()
    mr = collection.insert([vec])
    # ids = mr.primary_keys
    time_end = time.time()
    print("task {} cost time: {}".format(task_id, time_end - time_start))
    logging.info("task {}, process {}, insert number:{},insert time:{}".format(task_id, os.getpid(), len(ids),
                                                                               time_end - time_start))


def multi_insert_pool(collection_name):
    p = Pool(PROCESS_NUM)
    begin_time = time.time()
    loop = TOTAL_NUM // BATCH_SIZE
    print(loop)
    for i in range(loop):
        p.apply_async(sub_insert, (i, collection_name,))
    p.close()
    p.join()
    print("total cost time: {}".format(time.time() - begin_time))


def main():
    try:
        opts, args = getopt.getopt(
            sys.argv[1:],
            "hn:ci",
            ["help", "name=", "create", "insert"])
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
	# create a collection
        elif opt_name in ("-c", "--create"):
            create_collection(collection_name)
            sys.exit(2)
	# Insert data concurrently
        elif opt_name in ("-i", "--insert"):
            multi_insert_pool(collection_name)
            sys.exit(2)


if __name__ == "__main__":
    main()
