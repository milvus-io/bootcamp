import sys, getopt
import os
import time
from milvus import *
import numpy as np


QUERY_PATH = '/data/workspace/milvus_data/sift_data/bigann_query.bvecs'
# query_location = 0

MILVUS_TABLE = 'partition_query'


SERVER_ADDR = "0.0.0.0"
SERVER_PORT = 19534


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


def load_query_list(query_location):
    fname = QUERY_PATH
    x = np.memmap(fname, dtype='uint8', mode='r')
    d = x[:4].view('int32')[0]
    data =  x.reshape(-1, d + 4)[query_location:(query_location+1), 4:]
    data = (data + 0.5) / 255
    query_vec = data.tolist()
    return query_vec


def search_in_milvus(vector,partition_tag):
    time_start = time.time()
    status, results = milvus.search_vectors(MILVUS_TABLE, query_records=vector, top_k=10, nprobe=64, partition_tags=[partition_tag])
    time_end = time.time()
    if len(results) == 0:
        print("No vector satisfies the condition!")
        print("search time: ", time_end-time_start)
    else:
        print(results)
        print("search time: ", time_end-time_start)



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
            query_location = int(opt_value)
            query_vec = load_query_list(query_location)

        elif opt_name in ("-s", "--sex"):
            global sex_flag
            sex = opt_value
            sex_flag = True

        elif opt_name in ("-t", "--time"):
            time_insert = []
            global time_flag
            get_time = opt_value
            time_flag = True

        elif opt_name in ("-g", "--glasses"):
            global glasses_flag
            glasses = opt_value
            glasses_flag = True

        elif opt_name in ("-q", "--query"):
            connect_milvus_server()

            if sex_flag:
                if time_flag:
                    if glasses_flag:
                        partition_tag = get_time + "/" + sex + "/" + glasses
                        search_in_milvus(query_vec,partition_tag)
                    else:
                        partition_tag = get_time + "/" + sex + "/"
                        search_in_milvus(query_vec,partition_tag)
                else:
                    if glasses_flag:
                        partition_tag = "/" + sex + "/" + glasses
                        search_in_milvus(query_vec,partition_tag)
                    else:
                        partition_tag = "/" + sex + "/"
                        search_in_milvus(query_vec,partition_tag)
            else:
                if time_flag:
                    if glasses_flag:
                        partition_tag = get_time + "/.+" + glasses
                        search_in_milvus(query_vec,partition_tag)
                    else:
                        partition_tag =  glasses
                        search_in_milvus(query_vec,partition_tag)
                else:
                    if glasses_flag:
                        partition_tag =  get_time
                        search_in_milvus(query_vec,partition_tag)
                    else:
                        time_start = time.time()
                        status, results = milvus.search_vectors(MILVUS_TABLE, query_records=query_vec, top_k=10, nprobe=64)
                        time_end = time.time()
                        print(results)
                        print("search time: ", time_end-time_start)
                        
    

        else:
            print("wrong parameter")
            sys.exit(2)



if __name__ == "__main__":
    main(sys.argv[1:])
