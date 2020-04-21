from bert_serving.client import BertClient
from milvus import *
from functools import reduce
import numpy as np
import time
import src.pg_operating as pg_operating
import src.config as config
import logging
import traceback



milvus = Milvus()
bc = BertClient()


index_file_size = 1024
metric_type = MetricType.IP
nlist=16384


MILVUS_HOST = config.MILVUS_HOST
MILVUS_PORT = config.MILVUS_PORT

PG_HOST = config.PG_HOST
PG_PORT = config.PG_PORT
PG_USER = config.PG_USER
PG_PASSWORD = config.PG_PASSWORD
PG_DATABASE = config.PG_DATABASE


logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO)



def import_to_pg(table_name,ids,answer_file):
    conn = pg_operating.connect_postgres_server(PG_HOST, PG_PORT, PG_USER, PG_PASSWORD, PG_DATABASE)
    cur = conn.cursor()
    pg_operating.create_pg_table(conn, cur, table_name)
    pg_operating.record_txt(ids,answer_file)
    pg_operating.copy_data_to_pg(conn, cur, table_name)
    pg_operating.build_pg_index(conn, cur, table_name)


def normaliz_vec(vec_list):
    for i in range(len(vec_list)):
        vec = vec_list[i]
        square_sum = reduce(lambda x,y:x+y, map(lambda x:x*x ,vec))
        sqrt_square_sum = np.sqrt(square_sum)
        coef = 1/sqrt_square_sum
        vec = list(map(lambda x:x*coef, vec))
        vec_list[i] = vec
    return vec_list



def import_to_milvus(data,collection_name):
    vectors = bc.encode(data)
    question_vectors = normaliz_vec(vectors.tolist())
    status, ids = milvus.add_vectors(collection_name=collection_name, records=question_vectors)
    print(status)
    # index_param = {'index_type': IndexType.IVF_SQ8, 'nlist': nlist}
    # status = milvus.create_index(collection_name,index_param)
    # print(status)
    return ids


def create_milvus_table(collection_name):
    param = {'collection_name': collection_name, 'dimension': 768, 'index_file_size':index_file_size, 'metric_type':metric_type}
    status = milvus.create_table(param)
    print(status)
    # index_param = {'index_type': IndexType.IVF_SQ8, 'nlist': nlist}
    # milvus.create_index(collection_name,index_param)


def has_table(collection_name):
    status, ok = milvus.has_collection(collection_name)
    if not ok:
        # print("create table.")
        create_milvus_table(collection_name)
        index_param = {'nlist': nlist}
        status = milvus.create_index(collection_name,IndexType.IVF_SQ8,index_param)
        print(status)
    # print("insert into:", collection_name)


def connect_milvus_server():
    try:
        # milvus = Milvus()
        status = milvus.connect(MILVUS_HOST, MILVUS_PORT, timeout=30)
        logging.info(status)
    except Exception as e:
        logging.error(e)
        logging.error(traceback.format_exc())
        return None


def read_data(file_dir):
    data = []
    with open(file_dir,'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            if line:
                data.append(line)
    return data



def import_data(collection_name, question_dir, answer_dir):
    question_data = read_data(question_dir)
    connect_milvus_server()
    has_table(collection_name)
    ids = import_to_milvus(question_data,collection_name)
    import_to_pg(collection_name,ids,answer_dir)
        

def search_in_milvus(collection_name, query_sentence):
    logging.info("start test process ...")
    query_data = [query_sentence]
    try:
        vectors = bc.encode(query_data)
    except:
        return "bert service disconnect"
    query_list = normaliz_vec(vectors.tolist())
    #connect_milvus_server()
    try:
        status = milvus.connect(MILVUS_HOST, MILVUS_PORT, timeout=10)
        #logging.info(status)
    except:
        return "milvus service connection failed"
    try:
        logging.info("start search in milvus...")
        search_params = {'nprobe': 64}
        status,results = milvus.search_vectors(collection_name=collection_name, query_records=query_list, top_k=1, params=search_params)
        if results[0][0].distance < 0.9:
            return "对不起，我暂时无法为您解答该问题"
    except:
        return "milvus service disconnect"

    try:
        conn = pg_operating.connect_postgres_server(PG_HOST, PG_PORT, PG_USER, PG_PASSWORD, PG_DATABASE)
        cur = conn.cursor()
    except:
        return "Service connection failed"
    try:
        logging.info("start search in pg ...")
        rows = pg_operating.search_in_pg(conn, cur, results[0][0].id, collection_name)
        out_put = rows[0][1]
        return out_put
    except:
        return "Service disconnect"
    finally:
        if milvus:
            milvus.disconnect()
        conn.close()

