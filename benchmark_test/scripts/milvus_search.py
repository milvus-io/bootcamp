import sys, getopt
import numpy  as np
import time
import random
import os
from milvus import Milvus, Prepare, IndexType, Status

MILVUS = Milvus()
SERVER_ADDR = "0.0.0.0"
SERVER_PORT = 19530

NQ_FOLDER_NAME = 'query_data'
SE_FOLDER_NAME = 'search_output'
SE_FILE_NAME = '_output.txt'
BASE_FOLDER_NAME = 'bvecs_data/'
TOFILE = False
GT_NQ = 10000
CSV = False
UINT8 = False

NPROBE = 1

def connect_server():
    print("connect to milvus.")
    status = MILVUS.connect(host=SERVER_ADDR, port=SERVER_PORT)
    handle_status(status=status)
    return status


# the status of milvus
def handle_status(status):
    if status.code != Status.SUCCESS:
        print(status)
        sys.exit(2)


def load_all_vec():
    filenames = os.listdir(NQ_FOLDER_NAME)
    filenames.sort()
    for filename in filenames:
        filename = NQ_FOLDER_NAME + '/' + filename
        if CSV:
            import pandas as pd
            data = pd.read_csv(filename, header=None)
            data = np.array(data)
        else:
            data = np.load(filename)
        if UINT8:
            data = (data + 0.5) / 255
        vec_list = []
        for i in range(len(data)):
            vec_list.append(data[i].tolist())
    return vec_list


def save_re_to_file(table_name, rand, results, nprobe):
    if not os.path.exists(SE_FOLDER_NAME):
        os.mkdir(SE_FOLDER_NAME)
    file_name = SE_FOLDER_NAME + '/' + table_name + '_' + str(nprobe) + SE_FILE_NAME
    with open(file_name, 'w') as f:
        for i in range(len(results)):
            for j in range(len(results[i])):
                if rand is not None:
                    line = str(rand[i]) + ' ' + str(results[i][j].id) + ' ' + str(results[i][j].distance)
                else:
                    line = str(i) + ' ' + str(results[i][j].id) + ' ' + str(results[i][j].distance)
                f.write(line + '\n')
            f.write('\n')
    f.close()


def search_vec_list(table_name, nq, topk, nprobe):
    rand = None
    query_list = []
    vectors = load_all_vec()
    if nq != 0:
        try:
            rand = sorted(random.sample(range(0, GT_NQ), nq))
            for i in rand:
                query_list.append(vectors[i])
        except:
            print("Error: please change NQ as the num of query list")
            sys.exit()
    else:
        query_list = vectors
    print("table name:", table_name, "query list:", len(query_list), "topk:", topk, "nprobe:", nprobe)
    time_start = time.time()
    status, results = MILVUS.search_vectors(table_name=table_name, query_records=query_list, top_k=topk, nprobe=nprobe)
    time_end = time.time()
    time_cost = time_end - time_start
    print("time_search = ", time_cost)
    time_start = time.time()
    save_re_to_file(table_name, rand, results, nprobe)
    time_end = time.time()
    time_cost = time_end - time_start
    print("time_save = ", time_cost)


def get_file_loc_txt(table_name, nprobe):
    se_file = SE_FOLDER_NAME + '/' + table_name + '_' + nprobe + SE_FILE_NAME
    fnames_file = SE_FOLDER_NAME + '/' + table_name + '_file' + '_' + nprobe + SE_FILE_NAME
    filenames = os.listdir(BASE_FOLDER_NAME)
    filenames.sort()
    with open(se_file, 'r') as gt_f:
        with open(fnames_file, 'w') as fnames_f:
            for line in gt_f:
                if line != '\n':
                    data = line.split()
                    line = data[1]
                    print(line)
                    loca = int(line[1:5])
                    offset = int(line[5:11])
                    fnames_f.write(filenames[loca] + ' ' + str(offset + 1) + '\n')
                else:
                    fnames_f.write('\n')

def main():
    try:
        opts, args = getopt.getopt(
            sys.argv[1:],
            "hst:q:k:n:",
            ["help", "search", "table=", "nq=", "topk=", "nprobe="],
        )
    except getopt.GetoptError:
        print("Usage: test.py --table <table_name> [-q <nq>] -k <topk> -s")
        sys.exit(2)
    nq = 0
    nprobe = NPROBE
    for opt_name, opt_value in opts:
        if opt_name in ("-h", "--help"):
            print("test.py -table <table_name> [-q <nq>] -k <topk> -s")
            sys.exit()
        elif opt_name in ("-t", "--table"):
            table_name = opt_value
        elif opt_name in ("-q", "--nq"):
            nq = int(opt_value)
        elif opt_name in ("-k", "--topk"):
            topk = int(opt_value)
        elif opt_name in ("-n", "--nprobe"):
            nprobe = int(opt_value)
        elif opt_name == "-s":
            connect_server()
            search_vec_list(table_name, nq, topk, nprobe)  # test.py --table <tablename> [-q <nq>] -k <topk> [-n <nprobe>] -s
            if TOFILE:
                get_file_loc_txt(table_name, nprobe)
                print("get the search file's location success.")
            sys.exit()


if __name__ == '__main__':
    main()
