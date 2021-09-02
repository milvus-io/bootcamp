import os
import numpy as np
import time
from sklearn.preprocessing import normalize
from logs import LOGGER
from config import FILE_TYPE, BASE_FILE_PATH, IS_UINT8, IF_NORMALIZE, TOTAL_VECTOR_COUNT, IMPORT_CHUNK_SIZE, \
    METRIC_TYPE, NLIST, PQ_M, N_TREE, EFCONSTRUCTION, HNSW_M, SEARCH_LENGTH, OUT_DEGREE, CANDIDATE_POOL, KNNG


def load_csv_data(filename):
    import pandas as pd
    # filename = BASE_FILE_PATH + "/" + filename
    data = pd.read_csv(filename, header=None)
    data = np.array(data)
    if IS_UINT8:
        data = (data + 0.5) / 255
    if IF_NORMALIZE:
        data = normalize(data)
    data = data.tolist()
    return data


def csv_to_milvus(collection_name, client):
    filenames = os.listdir(BASE_FILE_PATH)
    filenames.sort()
    total_insert_time = 0
    for filename in filenames:
        fname = os.path.join(BASE_FILE_PATH, filename)
        vectors = load_csv_data(fname)
        collection_rows = client.count(collection_name)
        vectors_ids = [id for id in range(collection_rows, collection_rows + len(vectors))]
        time_add_start = time.time()
        ids = client.insert(collection_name, vectors, vectors_ids)
        total_insert_time = total_insert_time + time.time() - time_add_start
        print(filename, " insert time: ", time.time() - time_add_start)
    print("total insert time: ", total_insert_time)


def load_fvecs_data(base_len, idx, fname):
    begin_num = base_len * idx
    x = np.memmap(fname, dtype='uint8', mode='r')
    d = x[:4].view('int32')[0]
    data = x.view('float32').reshape(-1, d + 1)[begin_num:(begin_num + base_len), 1:]
    # data = (data+0.5)/255
    if IF_NORMALIZE:
        data = normalize(data)
    data = data.tolist()
    return data


def fvecs_to_milvus(collection_name, client):
    fname = BASE_FILE_PATH
    count = 0
    total_insert_time = 0
    while count < (TOTAL_VECTOR_COUNT // IMPORT_CHUNK_SIZE):
        vectors = load_fvecs_data(IMPORT_CHUNK_SIZE, count, fname)
        vectors_ids = [id for id in range(count * IMPORT_CHUNK_SIZE, (count + 1) * IMPORT_CHUNK_SIZE)]
        time_add_start = time.time()
        ids = client.insert(collection_name, vectors, vectors_ids)
        total_insert_time = total_insert_time + time.time() - time_add_start
        print(count * IMPORT_CHUNK_SIZE, (count + 1) * IMPORT_CHUNK_SIZE, 'time:',
              time.time() - time_add_start)
        count = count + 1
    print("total insert time: ", total_insert_time)


def load_npy_data(filename):
    data = np.load(filename)
    if IS_UINT8:
        data = (data + 0.5) / 255
    if IF_NORMALIZE:
        data = normalize(data)
    data = data.tolist()
    return data


def npy_to_milvus(collection_name, client):
    filenames = os.listdir(BASE_FILE_PATH)
    filenames.sort()
    total_insert_time = 0
    collection_rows = client.count(collection_name)
    for filename in filenames:
        vectors = load_npy_data(os.path.join(BASE_FILE_PATH, filename))
#         collection_rows = client.count(collection_name)
        vectors_ids = [id for id in range(collection_rows, collection_rows + len(vectors))]
        time_add_start = time.time()
        ids = client.insert(collection_name, vectors, vectors_ids)
        total_insert_time = total_insert_time + time.time() - time_add_start
        print(filename, "insert rows", len(ids), " insert milvus time: ", time.time() - time_add_start)
        collection_rows = collection_rows + len(ids)
    print("total insert time: ", total_insert_time)


def load_bvecs_data(base_len, idx, fname):
    begin_num = base_len * idx
    x = np.memmap(fname, dtype='uint8', mode='r')
    d = x[:4].view('int32')[0]
    data = x.reshape(-1, d + 4)[begin_num:(begin_num + base_len), 4:]
    data = (data + 0.5) / 255
    if IF_NORMALIZE:
        data = normalize(data)
    data = data.tolist()
    return data


def bvecs_to_milvus(collection_name, client):
    fname = BASE_FILE_PATH
    count = 0
    total_insert_time = 0
    collection_rows = client.count(collection_name)
    while count < (TOTAL_VECTOR_COUNT // IMPORT_CHUNK_SIZE):
        vectors = load_bvecs_data(IMPORT_CHUNK_SIZE, count, fname)
#         collection_rows = client.count(collection_name)
        vectors_ids = [id for id in range(collection_rows, collection_rows + len(vectors))]
        time_add_start = time.time()
        ids = client.insert(collection_name, vectors, vectors_ids)
        print(count * IMPORT_CHUNK_SIZE, (count + 1) * IMPORT_CHUNK_SIZE, 'time:',
              time.time() - time_add_start)
        total_insert_time = total_insert_time + time.time() - time_add_start
        count = count + 1
        collection_rows = collection_rows + len(ids)
    print("total insert time: {}".format(total_insert_time))


def insert_data(client, collection_name):
    if FILE_TYPE[0] == 'npy':
        npy_to_milvus(collection_name, client)
    if FILE_TYPE[0] == 'csv':
        csv_to_milvus(collection_name, client)
    if FILE_TYPE[0] == 'bvecs':
        bvecs_to_milvus(collection_name, client)
    if FILE_TYPE[0] == 'fvecs':
        fvecs_to_milvus(collection_name, client)


def get_index_params(index_type):
    if index_type == 'FLAT':
        index_param = {"index_type": index_type}
    elif index_type == 'RNSG':
        params = {"search_length": SEARCH_LENGTH, "out_degree": OUT_DEGREE, "candidate_pool_size": CANDIDATE_POOL,
                  "knng": KNNG}
        index_param = {"index_type": index_type, "metric_type": METRIC_TYPE, "params": params}
    elif index_type == 'HNSW':
        params = {"M": HNSW_M, "efConstruction": EFCONSTRUCTION}
        index_param = {"index_type": index_type, "metric_type": METRIC_TYPE, "params": params}
    elif index_type == 'ANNOY':
        params = {"n_tress": N_TREE}
        index_param = {"index_type": index_type, "metric_type": METRIC_TYPE, "params": params}
    elif index_type == 'IVF_PQ':
        params = {"nlist": NLIST, "m": PQ_M}
        index_param = {"index_type": index_type, "metric_type": METRIC_TYPE, "params": params}
    else:
        params = {"nlist": NLIST}
        index_param = {"index_type": index_type, "metric_type": METRIC_TYPE, "params": params}
    LOGGER.info(index_param)
    return index_param


def create_index(client, collection_name, index_type):
    index_param = get_index_params(index_type)
    time1 = time.time()
    client.create_index(collection_name, index_param)
    LOGGER.info("create index total cost time: {}".format(time.time() - time1))
