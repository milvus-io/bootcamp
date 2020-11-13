# import face_recognition
import os
import time
from milvus import Milvus, DataType
import numpy as np
import random




MILVUS_collection = 'mixe01'
PG_TABLE_NAME = 'mixe_query'

FILE_PATH = 'bigann_base.bvecs'

VEC_NUM = 100000
BASE_LEN = 100000

VEC_DIM = 128

SERVER_ADDR = "127.0.0.1"
SERVER_PORT = 19573



# milvus = Milvus()

def load_bvecs_data(fname,base_len,idx):
    begin_num = base_len * idx
    # print(fname, ": ", begin_num )
    x = np.memmap(fname, dtype='uint8', mode='r')
    d = x[:4].view('int32')[0]
    data =  x.reshape(-1, d + 4)[begin_num:(begin_num+base_len), 4:]   
    data = (data + 0.5) / 255
    # data = normaliz_data(data)
    data = data.tolist()
    return data


def create_milvus_collection(milvus):
    if not milvus.has_collection(MILVUS_collection):
        collection_name = MILVUS_collection
        collection_param = {
            "fields": [
                {"name": "sex", "type": DataType.INT32},
                {"name": "is_glasses", "type": DataType.INT32},
                {"name": "get_time","type":DataType.INT32},
                {"name": "Vec", "type": DataType.FLOAT_VECTOR, "params": {"dim": 128}},
            ],
            "segment_row_limit": 1000000,
            "auto_id": False
        }
        milvus.create_collection(collection_name,collection_param)

def build_collection(milvus):
    ivf_param = {"index_type": "IVF_SQ8", "metric_type": "L2", "params": {"nlist": 4096}}
    status = milvus.create_index(MILVUS_collection,"Vec",ivf_param)
    print(status)

def main():
    # connect_milvus_server()
    milvus = Milvus(host=SERVER_ADDR, port=SERVER_PORT)
    create_milvus_collection(milvus)
    build_collection(milvus)
    count = 0
    while count < (VEC_NUM // BASE_LEN):
        vectors = load_bvecs_data(FILE_PATH,BASE_LEN,count)
        vectors_ids = [id for id in range(count*BASE_LEN,(count+1)*BASE_LEN)]

        sex = [random.randint(0, 2) for _ in range(10000)]
        get_time = [random.randint(2017, 2020) for _ in range(10000)]
        is_glasses = [random.randint(10, 13) for _ in range(10000)]
        hybrid_entities = [
            {"name": "sex", "values": sex, "type": DataType.INT32},
            {"name": "is_glasses", "values": is_glasses, "type": DataType.INT32},
            {"name": "get_time","values": get_time, "type":DataType.INT32},
            {"name": "Vec", "values": vectors, "type": DataType.FLOAT_VECTOR}
        ]
        time_start = time.time()
        res = milvus.insert(MILVUS_collection, hybrid_entities, ids=vectors_ids)
        time_end = time.time()
        print(count, "insert milvue time: ", time_end-time_start)


if __name__ == '__main__':
    main()
