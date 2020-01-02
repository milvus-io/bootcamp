import getopt
import os
import sys
import time
from functools import reduce

import numpy as np
from milvus import *

SERVER_ADDR = "0.0.0.0"
SERVER_PORT = 19530

FILE_NPY_PATH = 'bvecs_data'
FILE_CSV_PATH = '/data/lym/dataset_test/csv_dataset'
FILE_FVECS_PATH = '/mnt/data/base.fvecs'
FILE_BVECS_PATH = '/data/workspace/milvus_data/sift_data/bigann_base.bvecs'

FVECS_VEC_NUM = 1000000
#FVECS_VEC_NUM = 1000000000
FVECS_BASE_LEN = 100000



milvus = Milvus()


is_uint8 = True
if_normaliz = False


def normaliz_data(vec_list):
    for i in range(len(vec_list)):
        vec = vec_list[i]
        square_sum = reduce(lambda x, y: x + y, map(lambda x: x * x, vec))
        sqrt_square_sum = np.sqrt(square_sum)
        coef = 1 / sqrt_square_sum
        vec = list(map(lambda x: x * coef, vec))
        vec_list[i] = vec
    return vec_list


def load_npy_data(filename):
    filename = FILE_NPY_PATH + "/" + filename
    data = np.load(filename) 
    if is_uint8:
        data = (data+0.5)/255
    if if_normaliz:
        data = normaliz_data(data)
    data = data.tolist()
    return data


def load_csv_data(filename):
    import pandas as pd
    filename = FILE_CSV_PATH + "/" + filename
    data = pd.read_csv(filename, header=None)
    data = np.array(data)
    if is_uint8:
        data = (data+0.5)/255
    if if_normaliz:
        data = normaliz_data(data)
    data = data.tolist()
    return data


def load_fvecs_data(fname, base_len, idx):
    begin_num = base_len * idx
    x = np.memmap(fname, dtype='uint8', mode='r')
    d = x[:4].view('int32')[0]
    data = x.view('float32').reshape(-1, d + 1)[begin_num:(begin_num + base_len), 1:]
    if is_uint8:
        data = (data+0.5)/255
    if if_normaliz:
        data = normaliz_data(data)
    data = data.tolist()
    return data


def load_bvecs_data(fname, base_len, idx):
    begin_num = base_len * idx
    # print(fname, ": ", begin_num)
    x = np.memmap(fname, dtype='uint8', mode='r')
    d = x[:4].view('int32')[0]
    data = x.reshape(-1, d + 4)[begin_num:(begin_num + base_len), 4:]
    if is_uint8:
        data = (data+0.5)/255
    if if_normaliz:
        data = normaliz_data(data)
    data = data.tolist()
    return data


def handle_status(status):
    if status.code != Status.SUCCESS:
        print(status)
        sys.exit(2)


def connect_milvus_server():
    print("connect to milvus")
    status = milvus.connect(host=SERVER_ADDR, port=SERVER_PORT, timeout=1000 * 1000 * 20)
    handle_status(status=status)
    return status



def npy_to_milvus(MILVUS_TABLE):
    filenames = os.listdir(FILE_NPY_PATH)
    filenames.sort()
    file_index = 0
    for filename in filenames:
        vectors = load_npy_data(filename)
        vectors_ids = []
        for i in range(len(vectors)):
            location = '8' + '%04d'%file_index  + '%06d'%i
            vectors_ids.append(int(location))
        time_add_start = time.time()
        status, ids = milvus.add_vectors(table_name=MILVUS_TABLE, records=vectors, ids=vectors_ids)
        time_add_end = time.time()
        print(filename, " insert milvus time: ", time_add_end - time_add_start)                
        file_index = file_index + 1

def csv_to_milvus(MILVUS_TABLE):
    filenames = os.listdir(FILE_CSV_PATH)
    filenames.sort()
    file_index = 0
    for filename in filenames:
        vectors = load_csv_data(filename)
        vectors_ids = []
        for i in range(len(vectors)):
            location = '8' + '%04d'%file_index  + '%06d'%i
            vectors_ids.append(int(location))
        time_add_start = time.time()
        status, ids = milvus.add_vectors(table_name=MILVUS_TABLE, records=vectors, ids=vectors_ids)
        time_add_end = time.time()
        print(filename, " insert time: ", time_add_end - time_add_start)
        file_index = file_index + 1



def main(argv):
    try:
        opts, args = getopt.getopt(
            sys.argv[1:],
            "ncfbt:",
            ["npy", "csv", "fvecs", "bvecs","table="],
        )
        # print(opts)
    except getopt.GetoptError:
        print("Usage: load_vec_to_milvus.py -n <npy>  -c <csv> -f <fvecs> -b <bvecs>")
        sys.exit(2)

    for opt_name, opt_value in opts:
        if opt_name in ("-t", "--table"):
            MILVUS_TABLE = opt_value
            PG_TABLE_NAME = opt_value
        elif opt_name in ("-n", "--npy"):
            connect_milvus_server()
            npy_to_milvus(MILVUS_TABLE)
        elif opt_name in ("-c", "--csv"):
            connect_milvus_server()
            csv_to_milvus(MILVUS_TABLE)
            
        elif opt_name in ("-f", "--fvecs"):
            connect_milvus_server()
            count = 0
            while count < (FVECS_VEC_NUM // FVECS_BASE_LEN):
                vectors = load_fvecs_data(FILE_FVECS_PATH, FVECS_BASE_LEN, count)
                print(count*FVECS_BASE_LEN, " ", (count+1)*FVECS_BASE_LEN)
                vectors_ids = [id for id in range(count*FVECS_BASE_LEN,(count+1)*FVECS_BASE_LEN)]                
                time_add_start = time.time()
                status, ids = milvus.add_vectors(table_name=MILVUS_TABLE, records=vectors, ids=vectors_ids)
                time_add_end = time.time()
                print(count, " insert to milvus time: ", time_add_end - time_add_start)
                count = count + 1
                
        elif opt_name in ("-b", "--bvecs"):
            connect_milvus_server()
            count = 0
            while count < (FVECS_VEC_NUM // FVECS_BASE_LEN):
                vectors = load_bvecs_data(FILE_BVECS_PATH, FVECS_BASE_LEN, count)
                print(count*FVECS_BASE_LEN, " ", (count+1)*FVECS_BASE_LEN)
                vectors_ids = [id for id in range(count*FVECS_BASE_LEN,(count+1)*FVECS_BASE_LEN)]                
                time_add_start = time.time()
                status, ids = milvus.add_vectors(table_name=MILVUS_TABLE, records=vectors, ids=vectors_ids)
                time_add_end = time.time()
                print(status,count, " insert to milvus time: ", time_add_end - time_add_start)
                count = count + 1
        else:
            print("wrong parameter")
            sys.exit(2)

if __name__ == "__main__":
    main(sys.argv[1:])
