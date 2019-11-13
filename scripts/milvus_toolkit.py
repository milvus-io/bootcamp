import time
import os
import getopt
import sys
import datetime
import numpy as np
from milvus import *

MILVUS = Milvus()
SERVER_ADDR = "0.0.0.0"
SERVER_PORT = 19530


index_file_size = 1024
metric_type = MetricType.L2
nlist=16384


NL_FOLDER_NAME = '/data/lcl/200_ann_test/source_data'
NQ_FOLDER_NAME = 'query_data'
#NQ_FOLDER_NAME = '/data/workspace/milvus_data/512_data/source_data'
PE_FOLDER_NAME = 'performance'


IS_CSV = False
IS_UINT8 = False
nq_scope = [1, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800]
topk_scope = [1,1, 20, 50, 100, 300, 500, 800, 1000]


PE_FILE_NAME = '_output.csv'
IDMAP_FOLDER_NAME = 'idmap'
IDMAP_NAME = '_idmap.txt'


# the status of milvus
def handle_status(status):
    if status.code != Status.SUCCESS:
        print(status)
        sys.exit(2)


# connect to the milvus server
def connect_server():
    status = MILVUS.connect(host=SERVER_ADDR, port=SERVER_PORT)
    print("connect to milvus")
    handle_status(status=status)
    return status


# -c/create the table with milvus
def create_table(table_name, dim):
    param = {'table_name': table_name, 'dimension': dim, 'index_file_size':index_file_size, 'metric_type':metric_type}
    print("create table: ", table_name, " dimension:", dim)
    return MILVUS.create_table(param)


def table_show():
    print(MILVUS.show_tables())


def describe_table(table_name):
    print(MILVUS.describe_table(table_name)[1])


def delete_table(table_name):
    print("delete table:", table_name)
    MILVUS.delete_table(table_name=table_name)


def build_table(table_name,index_type):
    index_param = {'index_type': index_type, 'nlist': nlist}
    status = MILVUS.create_index(table_name,index_param)
    print(status)


def show_server_version():
    print(MILVUS.server_version()[1])


def show_client_version():
    print(MILVUS.client_version())


def table_rows(table_name):
    print(table_name, 'has', MILVUS.get_table_row_count(table_name)[1], 'rows')


def load_nq_vec(nq):
    vectors = []
    length = 0
    filenames = os.listdir(NQ_FOLDER_NAME)
    filenames.sort()
    for filename in filenames:
        vec_list = load_vec_list(NQ_FOLDER_NAME + '/' + filename)
        length += len(vec_list)
        if length > nq:
            num = nq % len(vec_list)
            vec_list = vec_list[0:num]
        vectors += vec_list
        if len(vectors) == nq:
            return vectors


def load_vec_list(file_name, num=0):
    if IS_CSV:
        import pandas as pd
        data = pd.read_csv(file_name, header=None)
        data = np.array(data)
    else:
        data = np.load(file_name)
    if IS_UINT8:
        data = (data + 0.5) / 255
    vec_list = data.tolist()
    return vec_list


def search_vec_list(table_name,np):
    # random1 = datetime.datetime.now().strftime("%m%d%H%M")
    if not os.path.exists(PE_FOLDER_NAME):
        os.mkdir(PE_FOLDER_NAME)
    filename = PE_FOLDER_NAME + '/' + table_name + '_' + str(np) + PE_FILE_NAME
    file = open(filename, "w+")
    file.write('nq,topk,total_time,avg_time' + '\n')
    for nq in nq_scope:
        time_start = time.time()
        query_list = load_nq_vec(nq)
        time_end = time.time()
        print("load query:", len(query_list), "time_load = ", time_end - time_start)
        for k in topk_scope:
            time_start = time.time()
            status,result = MILVUS.search_vectors(table_name=table_name, query_records=query_list, top_k=k, nprobe=np)
            #print(status)
            time_end = time.time()
            time_cost = time_end - time_start
            line = str(nq) + ',' + str(k) + ',' + str(round(time_cost, 4)) + ',' + str(round(time_cost / nq, 4)) + '\n'
            file.write(line)
            print(nq, k, time_cost)
        file.write('\n')
    file.close()
    print("search_vec_list done !")


def is_normalized():
    filenames = os.listdir(NL_FOLDER_NAME)
    filenames.sort()
    vetors = load_vec_list(NL_FOLDER_NAME+'/'+filenames[0])
    for i in range(10):
        sqrt_sum = np.sum(np.power(vetors[i], 2))
        print(sqrt_sum)

def main():
    try:
        opts, args = getopt.getopt(
            sys.argv[1:],
            "chdsn",
            ["help", "table=", "dim=", "index=", "nq=", "show", "describe", "delete", "build", "drop_index", "server_version",
             "client_version", "rows", "normal", "nprobe=", "has", "desc_index"]
        )
    except getopt.GetoptError:
        print("Usage: python milvus_toolkit.py -q <nq> -k <topk> -c <table> -s")
        sys.exit(2)

    for opt_name, opt_value in opts:
        if opt_name in ("-h", "--help"):
            print("python milvus_toolkit.py test.py -q <nq> -k <topk> -c <table> -c -s")
            sys.exit()
        elif opt_name == "--table":
            table_name = opt_value
        elif opt_name == "--dim":
            dim = int(opt_value)
        elif opt_name == "--index":
            index_type = opt_value
            if index_type == 'flat':
                it = IndexType.FLAT
            elif index_type == 'ivf':
                it = IndexType.IVFLAT
            elif index_type == 'sq8':
                it = IndexType.IVF_SQ8
            elif index_type == 'sq8h':
                it = IndexType.IVF_SQ8H
            elif index_type == 'nsg':
                it = IndexType.NSG
        elif opt_name == "-c":
            connect_server()
            create_table(table_name, dim)
        elif opt_name == "--show":
            connect_server()
            table_show()
        elif opt_name in ("-n", "--normal"):
            is_normalized()
        elif opt_name == "--describe":
            connect_server()
            describe_table(table_name)
        elif opt_name == "--has":
            connect_server()
            print(MILVUS.has_table(table_name=table_name))
        elif opt_name in ("-d", "--delete"):
            connect_server()
            delete_table(table_name)
            if os.path.exists(IDMAP_FOLDER_NAME + table_name + IDMAP_NAME):
                os.remove(IDMAP_FOLDER_NAME + table_name + IDMAP_NAME)
        elif opt_name == "--build":
            connect_server()
            print(table_name," ",it)
            time1 = time.time()
            build_table(table_name,it)
            time2 = time.time()
            print("total cost time: ", time2-time1)
        elif opt_name == "--drop_index":
            connect_server()
            print(MILVUS.drop_index(table_name))
        elif opt_name == "--desc_index":
            connect_server()
            print(MILVUS.describe_index(table_name))
        elif opt_name == "--server_version":
            connect_server()
            show_server_version()
        elif opt_name == "--client_version":
            connect_server()
            show_client_version()
        elif opt_name == "--rows":  # test.py --table <tablename> --rows
            connect_server()
            table_rows(table_name)
        elif opt_name == "--nprobe":
            np = int(opt_value)
        elif opt_name == "-s":  # test.py --table <tablename> --nprobe <np> -s
            connect_server()
            search_vec_list(table_name,np)
            sys.exit()


if __name__ == '__main__':
    main()
