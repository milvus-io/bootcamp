import sys, getopt
import os
import time
from milvus import *
import psycopg2
import numpy as np


QUERY_PATH = 'bigann_query.bvecs'
# query_location = 0

MILVUS_TABLE = 'mixe_query'
PG_TABLE_NAME = 'mixe_query'


SERVER_ADDR = "0.0.0.0"
SERVER_PORT = 19530


PG_HOST = "192.168.1.10"
PG_PORT = 5432
PG_USER = "postgres"
PG_PASSWORD = "postgres"
PG_DATABASE = "postgres"

TOP_K = 10
DISTANCE_THRESHOLD = 1


milvus = Milvus()


sex_flag = False
time_flag = False
glasses_flag = False

def handle_status(status):
    if status.code != Status.SUCCESS:
        print(status)
        sys.exit(2)

def connect_milvus_server():
    print("connect to milvus")
    status =  milvus.connect(host=SERVER_ADDR, port=SERVER_PORT,timeout = 1000 * 1000 * 20 )
    handle_status(status=status)
    return status


def connect_postgres_server():
    try:
        conn = psycopg2.connect(host=PG_HOST,port=PG_PORT,user=PG_USER,password=PG_PASSWORD,database=PG_DATABASE)
        print("connect the database!")
        return conn
    except:
        print ("unable to connect to the database")


def load_query_list(fname, query_location):
    query_location = int(query_location)
    x = np.memmap(fname, dtype='uint8', mode='r')
    d = x[:4].view('int32')[0]
    data =  x.reshape(-1, d + 4)[query_location:(query_location+1), 4:]
    data = (data + 0.5) / 255
    query_vec = data.tolist()
    return query_vec


def search_in_milvus(vector):
    output_ids = []
    output_distance = []
    status, results = milvus.search_vectors(table_name = MILVUS_TABLE,query_records=vector, top_k=TOP_K, nprobe=64)
    for result in results:
        # print(result)
        for i in range(TOP_K):
            if result[i].distance < DISTANCE_THRESHOLD:
                output_ids.append(result[i].id)
                output_distance.append(result[i].distance)
    # print(output_ids)
    return  output_ids,output_distance


def merge_rows_distance(rows,ids,distance):
    new_results = []
    if len(rows)>0:
        for row in rows:
            index_flag = ids.index(row[0])
            temp = [row[0]] + list(row[2:5]) + [distance[index_flag]]
            new_results.append(temp)
        new_results = np.array(new_results)
        sort_arg = np.argsort(new_results[:,4])
        new_results = new_results[sort_arg].tolist()
        print("\nids                      sex        time                        glasses  distance")
        for new_result in new_results:
            print( new_result[0], "\t", new_result[1], new_result[2], "\t", new_result[3], "\t", new_result[4])
    else:
        print("no result")

def search_in_pg_0(conn,cur,result_ids,result_distance,sex,time,glasses):
    sql1 = str(result_ids[0])
    i = 1
    while i < len(result_ids):
        sql1 = sql1 + "," + str(result_ids[i])
        i = i + 1
    sql = "select * from " + PG_TABLE_NAME + " where ids in (" + sql1 + ") and sex='" + sex + "' and get_time between '" + time[0] + "' and '" + time[1] + "' and is_glasses='" + str(glasses) + "';"
    # print(sql)

    try:
        cur.execute(sql)
        rows=cur.fetchall()
        # print("search sucessful!")
        print(len(rows))
        return rows
    except:
        print("search faild!")

def search_in_pg_1(conn,cur,result_ids,result_distance,sex,time):
    sql1 = str(result_ids[0])
    i = 1
    while i < len(result_ids):
        sql1 = sql1 + "," + str(result_ids[i])
        i = i + 1
    sql = "select * from " + PG_TABLE_NAME + " where ids in (" + sql1 + ") and sex='" + sex + "' and get_time between '" + time[0] + "' and '" + time[1] + "';"
    # print(sql)

    try:
        cur.execute(sql)
        rows=cur.fetchall()
        # print("search sucessful!")
        print(len(rows))
        return rows
    except:
        print("search faild!")


def search_in_pg_2(conn,cur,result_ids,result_distance,sex,glasses):
    sql1 = str(result_ids[0])
    i = 1
    while i < len(result_ids):
        sql1 = sql1 + "," + str(result_ids[i])
        i = i + 1
    sql = "select * from " + PG_TABLE_NAME + " where ids in (" + sql1 + ") and sex='" + sex + "' and is_glasses='" + str(glasses) + "';"
    # print(sql)

    try:
        cur.execute(sql)
        rows=cur.fetchall()
        # print("search sucessful!")
        print(len(rows))
        return rows
    except:
        print("search faild!")


def search_in_pg_3(conn,cur,result_ids,result_distance,sex):
    sql1 = str(result_ids[0])
    i = 1
    while i < len(result_ids):
        sql1 = sql1 + "," + str(result_ids[i])
        i = i + 1
    sql = "select * from " + PG_TABLE_NAME + " where ids in (" + sql1 + ") and sex='" + sex + "';"
    # print(sql)

    try:
        cur.execute(sql)
        rows=cur.fetchall()
        # print("search sucessful!")
        print(len(rows))
        return rows
    except:
        print("search faild!")


def search_in_pg_4(conn,cur,result_ids,result_distance,time,glasses):
    sql1 = str(result_ids[0])
    i = 1
    while i < len(result_ids):
        sql1 = sql1 + "," + str(result_ids[i])
        i = i + 1
    sql = "select * from " + PG_TABLE_NAME + " where ids in (" + sql1 + ") and get_time between '" + time[0] + "' and '" + time[1] + "' and is_glasses='" + str(glasses) + "';"
    # print(sql)

    try:
        cur.execute(sql)
        rows=cur.fetchall()
        # print("search sucessful!")
        print(len(rows))
        return rows
    except:
        print("search faild!")


def search_in_pg_5(conn,cur,result_ids,result_distance,time):
    sql1 = str(result_ids[0])
    i = 1
    while i < len(result_ids):
        sql1 = sql1 + "," + str(result_ids[i])
        i = i + 1
    # print(time[0])
    # print(time[1])
    sql = "select * from " + PG_TABLE_NAME + " where ids in (" + sql1 + ") and get_time between '" + time[0] + "' and '" + time[1] + "';"
    # print(sql)
    try:
        cur.execute(sql)
        rows=cur.fetchall()
        # print("search sucessful!")
        print(len(rows))
        return rows
    except:
        print("search faild!")


def search_in_pg_6(conn,cur,result_ids,result_distance,glasses):
    sql1 = str(result_ids[0])
    i = 1
    while i < len(result_ids):
        sql1 = sql1 + "," + str(result_ids[i])
        i = i + 1
    sql = "select * from " + PG_TABLE_NAME + " where ids in (" + sql1 + ") and is_glasses='" + str(glasses) + "';"
    # print(sql)

    try:
        cur.execute(sql)
        rows=cur.fetchall()
        # print("search sucessful!")
        print(len(rows))
        return rows
    except:
        print("search faild!")

def search_in_pg_7(conn,cur,result_ids,result_distance):
    sql1 = str(result_ids[0])
    i = 1
    while i < len(result_ids):
        sql1 = sql1 + "," + str(result_ids[i])
        i = i + 1
    sql = "select * from " + PG_TABLE_NAME + " where ids in (" + sql1 + ");"
    # print(sql)

    try:
        cur.execute(sql)
        rows=cur.fetchall()
        # print("search success!")
        print(len(rows))
        j = 0
        for row in rows:
            print(row[0], " ", row[1], " ", row[2], " ", row[3], " ", result_distance[j])
            j = j + 1
    except:
        print("search faild!")


def search_vecs_pg(conn,cur,id):
    sql = "select vecs from " + PG_TABLE_NAME + " where ids = " + id + ";"
    try:
        cur.execute(sql)
        rows=cur.fetchall()
        print(rows)
    except:
        print("search faild!")


def main(argv):
    try:
        opts, args = getopt.getopt(
            sys.argv[1:],
            "n:s:t:g:v:q",
            ["num=", "sex=", "time=", "glasses=", "query","vector="],
        )
        # print(opts)
    except getopt.GetoptError:
        print("Usage: load_vec_to_milvus.py -n <npy>  -c <csv> -f <fvecs> -b <bvecs>")
        sys.exit(2)

    for opt_name, opt_value in opts:
        if opt_name in ("-n", "--num"):
            query_location = opt_value
            query_vec = load_query_list(QUERY_PATH,query_location)

        elif opt_name in ("-s", "--sex"):
            global sex_flag
            sex = opt_value
            sex_flag = True

        elif opt_name in ("-t", "--time"):
            time_insert = []
            global time_flag
            temp = opt_value
            time_insert.append(temp[1:20])
            time_insert.append(temp[22:41])
            time_flag = True

        elif opt_name in ("-g", "--glasses"):
            global glasses_flag
            glasses = opt_value
            glasses_flag = True

        elif opt_name in ("-q", "--query"):
            connect_milvus_server()
            time_start_0 = time.time()
            result_ids, result_distance = search_in_milvus(query_vec)
            time_end_0 = time.time()            
            print("search in milvus cost time: ", time_end_0 - time_start_0)
            # print(len(result_ids))
            # print(result_ids)
            # print(result_distance)
            conn = connect_postgres_server()
            cur = conn.cursor()
            # print("begin!")
            # print(sex_flag, glasses_flag,time_flag)
            if len(result_ids)>0:
                if sex_flag:
                    if time_flag:
                        if glasses_flag:
                            # print(time[0])
                            # print(time[1])
                            time_start_1 = time.time()
                            rows = search_in_pg_0(conn,cur,result_ids, result_distance, sex,time_insert,glasses)
                            time_end_1 = time.time()
                            print("search in pg cost time: ", time_end_1 - time_start_1)
                            merge_rows_distance(rows,result_ids,result_distance)
                        else:
                            time_start_1 = time.time()
                            rows = search_in_pg_1(conn,cur,result_ids, result_distance, sex,time_insert)
                            time_end_1 = time.time()
                            print("search in pg cost time: ", time_end_1 - time_start_1)
                            merge_rows_distance(rows,result_ids,result_distance)
                    else:
                        if glasses_flag:
                            time_start_1 = time.time()
                            rows = search_in_pg_2(conn,cur,result_ids, result_distance, sex, glasses)
                            time_end_1 = time.time()
                            print("search in pg cost time: ", time_end_1 - time_start_1)
                            merge_rows_distance(rows,result_ids,result_distance)
                        else:
                            time_start_1 = time.time()
                            rows = search_in_pg_3(conn,cur,result_ids, result_distance,sex)
                            time_end_1 = time.time()
                            print("search in pg cost time: ", time_end_1 - time_start_1)
                            merge_rows_distance(rows,result_ids,result_distance)
                else:
                    if time_flag:
                        if glasses_flag:
                            time_start_1 = time.time()
                            rows = search_in_pg_4(conn,cur,result_ids,result_distance,time_insert,glasses)
                            time_end_1 = time.time()
                            print("search in pg cost time: ", time_end_1 - time_start_1)
                            merge_rows_distance(rows,result_ids,result_distance)
                        else:
                            time_start_1 = time.time()
                            rows = search_in_pg_5(conn,cur,result_ids,result_distance,time_insert)
                            time_end_1 = time.time()
                            print("search in pg cost time: ", time_end_1 - time_start_1)
                            merge_rows_distance(rows,result_ids,result_distance)
                    else:
                        if glasses_flag:
                            time_start_1 = time.time()
                            rows = search_in_pg_6(conn,cur,result_ids,result_distance,glasses)
                            time_end_1 = time.time()
                            print("search in pg cost time: ", time_end_1 - time_start_1)
                            merge_rows_distance(rows,result_ids,result_distance)
                        else:
                            time_start_1 = time.time()
                            search_in_pg_7(conn,cur,result_ids,result_distance)
                            time_end_1 = time.time()
                            print("search in pg cost time: ", time_end_1 - time_start_1)
                sys.exit(2)
            else:
                print("no vectors!")

        elif opt_name in ("-v", "--vector"):
            id = opt_value
            conn = connect_postgres_server()
            cur = conn.cursor()
            search_vecs_pg(conn,cur,id)
            sys.exit(2)

        else:
            print("wrong parameter")
            sys.exit(2)



if __name__ == "__main__":
    main(sys.argv[1:])
