from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import numpy as np
from multiprocessing import Process
from multiprocessing import Pool, cpu_count
import time
import logging
import os

logging.basicConfig(filename='benchmark.log', level=logging.DEBUG)


MILVUS_HOST = "127.0.0.1"
MILVUS_PORT = 19530

DIM = 768
TOTAL_NUM = 100000
BATCH_NUM = 10000
PROCESS_NUM = 5


def create_collection(collection_name):
    try:
        if not utility.has_collection(collection_name):
            field1 = FieldSchema(name="id", dtype=DataType.INT64, descrition="int64", is_primary=True, auto_id=True)
            field2 = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, descrition="float vector", dim=DIM,
                                 is_primary=False)
            schema = CollectionSchema(fields=[field1, field2], description="collection description")
            collection = Collection(name=collection_name, schema=schema)
            print("Create Milvus collection: {}".format(collection))
            return collection
        else:
            collection = Collection(collection_name)
            return collection
    except Exception as e:
        logging.error("Failed to create milvus collection: {}".format(e))
        # sys.exit(1)


def bvecs_mmap(fname):
    x = np.memmap(fname, dtype='uint8', mode='r')
    d = x[:4].view('int32')[0]
    return x.reshape(-1, d + 4)[:, 4:]


def read_bvecs_data(base_len, idx, fname):
    begin_num = base_len * idx
    x = np.memmap(fname, dtype='uint8', mode='r')
    d = x[:4].view('int32')[0]
    data = x.reshape(-1, d + 4)[begin_num:(begin_num + base_len), 4:]
    data = (data + 0.5) / 255
    if IF_NORMALIZE:
        data = normalize(data)
    data = data.tolist()
    return data


def sub_insert(vec, collection, task_id):
    print(task_id)
    # print("sub process {} insert begin".format(os.getpid()))
    time_start = time.time()
    mr = collection.insert([vec])
    ids = mr.primary_keys
    time_end = time.time()
    print("task {} cost time: {}".format(task_id, time_end-time_start))
    logging.info("insert number:{},insert time:{}".format(len(ids), time_end - time_start))
    return ids


def multi_insert(collection_name):
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
    collection = create_collection(collection_name)
    loop = TOTAL_NUM // (BATCH_NUM * PROCESS_NUM)
    for i in range(loop):
        print("loop: {}, begin.".format(i))
        thread_list = []
        vec = np.random.random((PROCESS_NUM, BATCH_NUM, DIM)).tolist()
        begin_time = time.time()
        for j in range(PROCESS_NUM):
            p = Process(target=sub_insert, args=(vec[j], collection, j))
            thread_list.append(p)
            p.start()
        for p in thread_list:
            p.join()
        print("loop {}: , total insert: {}, cost time: {}".format(i, PROCESS_NUM * BATCH_NUM, time.time() - begin_time))



if __name__ == "__main__":
    # multi_insert_pool('test8')
    multi_insert('test')
