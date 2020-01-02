# import face_recognition
import os
import time
from milvus import *
import psycopg2
import numpy as np
import random
from faker import Faker

fake = Faker()





MILVUS_TABLE = 'mixe_query'
PG_TABLE_NAME = 'mixe_query'

FILE_PATH = 'bigann_base.bvecs'

VEC_NUM = 100000000 
BASE_LEN = 100000

VEC_DIM = 128

SERVER_ADDR = "0.0.0.0"
SERVER_PORT = 19530

PG_HOST = "192.168.1.10"
PG_PORT = 5432
PG_USER = "postgres"
PG_PASSWORD = "postgres"
PG_DATABASE = "postgres"

milvus = Milvus()

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

def handle_status(status):
    if status.code != Status.SUCCESS:
        print(status)
        sys.exit(2)

def connect_milvus_server():
    print("connect to milvus")
    status =  milvus.connect(host=SERVER_ADDR, port=SERVER_PORT,timeout = 1000 * 1000 * 20 )
    handle_status(status=status)
    return status


def create_milvus_table():
    if not milvus.has_table(MILVUS_TABLE)[1]:
        param = {
            'table_name': MILVUS_TABLE,
            'dimension': VEC_DIM,
            'index_file_size':1024,
            'metric_type':MetricType.L2
        }
        milvus.create_table(param)

def build_table():
    index_param = {'index_type': IndexType.IVF_SQ8H, 'nlist': 16384}
    status = milvus.create_index(MILVUS_TABLE,index_param)
    print(status)

def connect_postgres_server():
    try: 
        conn = psycopg2.connect(host=PG_HOST, port=PG_PORT, user=PG_USER, password=PG_PASSWORD,database=PG_DATABASE)
        return conn
    except:
        print ("unable to connect to the database")


def create_pg_table(conn,cur):
    try:       
        sql = "CREATE TABLE " + PG_TABLE_NAME + " (ids bigint, sex char(10), get_time timestamp, is_glasses boolean);"
        cur.execute(sql)
        conn.commit()
        print("create postgres table!")
    except:
        print("can't create postgres table")


def insert_data_to_pg(ids, vector, sex, get_time, is_glasses, conn, cur):
    sql = "INSERT INTO " + PG_TABLE_NAME + " VALUES(" + str(ids) + ", array" + str(vector) + ", '" + str(sex) + "', '" + str(get_time) + "', '" + str(is_glasses) + "');"
    # print(sql)
    try:       
        # print(sql)
        cur.execute(sql)
        conn.commit()
        # print("insert success!")
    except:
        print("faild insert")


def copy_data_to_pg(conn, cur):
    # fname = './temp.csv'
    sql = "copy " + PG_TABLE_NAME + " from " + "'/data/lym/mixe_query/temp.csv'" + " with CSV delimiter '|';"
    # print(sql)
    try:
        cur.execute(sql)
        conn.commit()
        print("insert pg sucessful!")
    except:
        print("faild  copy!")


def build_pg_index(conn,cur):
    try:
        sql = "CREATE INDEX index_ids on " + PG_TABLE_NAME + "(ids);"
        cur.execute(sql)
        conn.commit()
        print("build index sucessful!")
    except:
        print("faild build index")


def record_txt(ids):
    fname = 'temp.csv'
    with open(fname,'w+') as f:
        for i in range(len(ids)):
            sex = random.choice(['female','male'])
            get_time = fake.past_datetime(start_date="-120d", tzinfo=None)
            is_glasses = random.choice(['True','False'])
            line = str(ids[i]) + "|" + sex + "|'" + str(get_time) + "'|" + str(is_glasses) + "\n"
            f.write(line)



def main():
    connect_milvus_server()
    create_milvus_table()
    build_table()
    conn = connect_postgres_server()
    cur = conn.cursor()
    create_pg_table(conn,cur)
    count = 0
    while count < (VEC_NUM // BASE_LEN):
        vectors = load_bvecs_data(FILE_PATH,BASE_LEN,count)
        vectors_ids = [id for id in range(count*BASE_LEN,(count+1)*BASE_LEN)]
        time_start = time.time()    
        status, ids = milvus.add_vectors(table_name=MILVUS_TABLE, records=vectors, ids=vectors_ids)
        time_end = time.time()
        print(count, "insert milvue time: ", time_end-time_start)
        # print(count)
        time_start = time.time()
        record_txt(ids)
        copy_data_to_pg(conn, cur)
        time_end = time.time()
        print(count, "insert pg time: ", time_end-time_start)

        count = count + 1

    build_pg_index(conn,cur)


if __name__ == '__main__':
    main()
