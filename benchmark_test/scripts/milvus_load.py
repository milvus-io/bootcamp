import getopt
import os
import sys
import time
from functools import reduce
import numpy as np
from milvus import *

import config



def connect_server():
    try:
        milvus = Milvus(host=config.MILVUS_HOST, port=config.MILVUS_PORT)
        return milvus
    except Exception as e:
        logging.error(e)



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
    filename = config.FILE_NPY_PATH + "/" + filename
    data = np.load(filename) 
    if config.IS_UINT8:
        data = (data+0.5)/255
    if config.if_normaliz:
        data = normaliz_data(data)
    data = data.tolist()
    return data


def load_csv_data(filename):
    import pandas as pd
    filename = config.FILE_CSV_PATH + "/" + filename
    data = pd.read_csv(filename, header=None)
    data = np.array(data)
    if config.IS_UINT8:
        data = (data+0.5)/255
    if config.if_normaliz:
        data = normaliz_data(data)
    data = data.tolist()
    return data


def load_bvecs_data(base_len, idx):
    fname = config.FILE_BVECS_PATH
    begin_num = base_len * idx
    # print(fname, ": ", begin_num)
    x = np.memmap(fname, dtype='uint8', mode='r')
    d = x[:4].view('int32')[0]
    data = x.reshape(-1, d + 4)[begin_num:(begin_num + base_len), 4:]
    data = (data+0.5)/255
    if config.if_normaliz:
        data = normaliz_data(data)
    data = data.tolist()
    return data


def load_fvecs_data(base_len, idx):
    fname = config.FILE_FVECS_PATH
    begin_num = base_len * idx
    x = np.memmap(fname, dtype='uint8', mode='r')
    d = x[:4].view('int32')[0]
    data = x.view('float32').reshape(-1, d + 1)[begin_num:(begin_num + base_len), 1:]
    # data = (data+0.5)/255
    if config.if_normaliz:
        data = normaliz_data(data)
    data = data.tolist()
    return data



def npy_to_milvus(collection_name,collection_rows,milvus):
    filenames = os.listdir(config.FILE_NPY_PATH)
    filenames.sort()
    # file_index = 0
    total_insert_time = 0
    for filename in filenames:
        vectors = load_npy_data(filename)
        vectors_ids = [id for id in range(collection_rows,collection_rows+len(vectors))]
        time_add_start = time.time()
        #status, ids = milvus.insert(collection_name=collection_name, records=vectors, ids=vectors_ids)
        status, ids = milvus.insert(collection_name=collection_name, records=vectors,ids=vectors_ids)
        # time_add_end = time.time()
        total_insert_time = total_insert_time + time.time() - time_add_start
        print(filename, " insert milvus time: ", time.time() - time_add_start)                
        collection_rows = collection_rows+len(vectors)
    print("total insert time: ", total_insert_time)



def csv_to_milvus(collection_name,collection_rows,milvus):
    filenames = os.listdir(config.FILE_CSV_PATH)
    filenames.sort()
    total_insert_time = 0
    for filename in filenames:
        vectors = load_csv_data(filename)
        vectors_ids = [id for id in range(collection_rows,collection_rows+len(vectors))]
        time_add_start = time.time()
        status, ids = milvus.insert(collection_name=collection_name, records=vectors, ids=vectors_ids)
        total_insert_time = total_insert_time + time.time() - time_add_start
        print(filename, " insert time: ", time.time() - time_add_start)
        collection_rows = collection_rows+len(vectors)
    print("total insert time: ", total_insert_time)



def bvecs_to_milvus(collection_name,milvus):
    count = 0
    total_insert_time = 0
    while count < (config.VECS_VEC_NUM // config.VECS_BASE_LEN):
        vectors = load_bvecs_data(config.VECS_BASE_LEN, count)
        #vectors_ids = [id for id in range(count*config.VECS_BASE_LEN,(count+1)*config.VECS_BASE_LEN)]                
        collection_rows = milvus.count_entities(collection_name)[1]
        vectors_ids = [id for id in range(collection_rows,collection_rows+len(vectors))]
        time_add_start = time.time()
        status, ids = milvus.insert(collection_name=collection_name, records=vectors, ids=vectors_ids)
        print(status,count*config.VECS_BASE_LEN,(count+1)*config.VECS_BASE_LEN,'time:', time.time() - time_add_start)
        total_insert_time = total_insert_time + time.time() - time_add_start
        count = count + 1
    print("total insert time: ", total_insert_time)



def fvecs_to_milvus(collection_name,milvus):
    count = 0
    total_insert_time = 0
    while count < (config.VECS_VEC_NUM // config.VECS_BASE_LEN):
        vectors = load_fvecs_data(config.VECS_BASE_LEN, count)
        vectors_ids = [id for id in range(count*config.VECS_BASE_LEN,(count+1)*config.VECS_BASE_LEN)]                
        time_add_start = time.time()
        status, ids = milvus.insert(collection_name=collection_name, records=vectors, ids=vectors_ids)
        total_insert_time = total_insert_time + time.time() - time_add_start
        print(status,count*config.VECS_BASE_LEN,(count+1)*config.VECS_BASE_LEN,'time:', time.time() - time_add_start)
        count = count + 1
    print("total insert time: ", total_insert_time)



def load(collection_name):
    milvus = connect_server()
    collection_rows = milvus.count_entities(collection_name)[1]
    file_type = config.FILE_TYPE
    if file_type == 'npy':
        npy_to_milvus(collection_name,collection_rows,milvus)
    if file_type == 'csv':
        csv_to_milvus(collection_name,collection_rows,milvus)
    if file_type == 'bvecs':
        bvecs_to_milvus(collection_name,milvus)
    if file_type == 'fvecs':
        fvecs_to_milvus(collection_name,milvus)






