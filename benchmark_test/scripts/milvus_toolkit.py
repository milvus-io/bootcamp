import time
import os
import getopt
import sys
import datetime
import numpy as np
from milvus import *
import config
import logging
import random


def connect_server():
    try:
        milvus = Milvus(host=config.MILVUS_HOST, port=config.MILVUS_PORT)
        return milvus
    except Exception as e:
        logging.error(e)


def build_collection(collection_name, it):
    milvus = connect_server()
    if it == 'flat':
        index_type = IndexType.FLAT
        index_param = {'nlist': config.NLIST}
    elif it == 'ivf_flat':
        index_type = IndexType.IVF_FLAT
        index_param = {'nlist': config.NLIST}
    elif it == 'sq8':
        index_type = IndexType.IVF_SQ8
        index_param = {'nlist': config.NLIST}
    elif it == 'sq8h':
        index_type = IndexType.IVF_SQ8H
        index_param = {'nlist': config.NLIST}
    elif it == 'pq':
        index_type = IndexType.IVF_PQ
        index_param = {'nlist': config.NLIST, 'm': config.PQ_M}
    elif it == 'nsg':
        index_type = IndexType.RNSG
        index_param = {'search_length': config.SEARCH_LENGTH, 'out_degree': config.OUT_DEGREE,
                       'candidate_pool_size': config.CANDIDATE_POOL, 'knng': config.KNNG}
    elif it == 'hnsw':
        index_type = IndexType.HNSW
        index_param = {'M': config.HNSW_M, 'efConstruction': config.EFCONSTRUCTION}
    else:
        print("error index_type, only support these index: flat, ivf_flat, sq8, sq8h, pq, nsg, hnsw")
        print("please try again!")
        sys.exit(2)

    print(collection_name, " ", index_type, " ", index_param)
    status = milvus.create_index(collection_name, index_type, index_param)
    print(status)


def search(collection_name, search_param):
    milvus = connect_server()
    performance_file = config.PERFORMANCE_RESULTS_PATH
    nq_scope = config.NQ_SCOPE
    topk_scope = config.TOPK_SCOPE
    if not os.path.exists(performance_file):
        os.mkdir(performance_file)
    filename = performance_file + '/' + collection_name + '_' + str(search_param) + '_performance.csv'
    search_params = get_search_params(collection_name, search_param, milvus)
    with open(filename, 'w+') as f:
        f.write("nq,topk,total_time,avg_time" + '\n')
        for nq in nq_scope:
            time_start = time.time()
            query_list = load_nq_vec(nq)
            print("load query:", len(query_list), "time_load = ", time.time() - time_start)
            for topk in topk_scope:
                time_start = time.time()
                status, result = milvus.search(collection_name=collection_name, query_records=query_list, top_k=topk,
                                               params=search_params)
                time_cost = time.time() - time_start
                print(nq, topk, time_cost)
                line = str(nq) + ',' + str(topk) + ',' + str(round(time_cost, 4)) + ',' + str(
                    round(time_cost / nq, 4)) + '\n'
                f.write(line)
            f.write('\n')
    # file.close()
    print("search_vec_list done !")


def get_search_params(collection_name, search_param, milvus):
    index_type = str(milvus.get_index_info(collection_name)[1]._index_type)
    if index_type == 'RNSG':
        search_params = {'search_length': search_param}
    elif index_type == 'HNSW':
        search_params = {'ef': search_param}
    else:
        search_params = {'nprobe': search_param}
    return search_params


def load_nq_vec(nq):
    vectors = []
    length = 0
    filenames = os.listdir(config.QUERY_FILE_PATH)
    filenames.sort()
    for filename in filenames:
        vec_list = load_vec_list(config.QUERY_FILE_PATH + '/' + filename)
        length += len(vec_list)
        if length > nq:
            num = nq % len(vec_list)
            vec_list = vec_list[0:num]
        vectors += vec_list
        if len(vectors) == nq:
            return vectors


def load_vec_list(file_name):
    if config.IS_CSV:
        import pandas as pd
        data = pd.read_csv(file_name, header=None)
        data = np.array(data)
    else:
        data = np.load(file_name)
    vec_list = data.tolist()
    return vec_list


def recall_test(collection_name, search_param):
    milvus = connect_server()
    vectors = load_vec_list(config.RECALL_QUERY_FILE)
    # for nq in config.nq_scope:
    nq = config.RECALL_NQ
    query_list = []
    rand = sorted(random.sample(range(0, len(vectors)), nq))
    for i in rand:
        query_list.append(vectors[i])
    # print("load query:", len(query_list))
    # rand=[0,1,2,3,4,5,6,7,8,9]
    search_params = get_search_params(collection_name, search_param, milvus)
    print("collection name:", collection_name, "query list:", len(query_list), "topk:", config.RECALL_TOPK,
          "search_params:", search_params)
    time_start = time.time()
    status, results = milvus.search(collection_name=collection_name, query_records=query_list, top_k=config.RECALL_TOPK,
                                    params=search_params)
    # time_end = time.time()
    time_cost = time.time() - time_start
    print("time_search = ", time_cost)
    save_re_to_file(collection_name, rand, results, search_param, nq)
    compute_recall(collection_name, nq, results, search_param, rand)


def save_re_to_file(collection_name, rand, results, search_param, nq):
    if not os.path.exists(config.recall_res_fname):
        os.mkdir(config.recall_res_fname)
    file_name = config.recall_res_fname + '/' + collection_name + '_' + str(search_param) + '_' + str(
        nq) + '_recall.txt'
    with open(file_name, 'w') as f:
        for i in range(len(results)):
            for j in range(len(results[i])):
                line = str(rand[i]) + ' ' + str(results[i][j].id) + ' ' + str(results[i][j].distance)
                f.write(line + '\n')
            f.write('\n')
    f.close()


def compute_recall(collection_name, nq, results, search_param, rand):
    ids = []
    # dis = []
    for nq_result in (results):
        temp = []
        for result in (nq_result):
            temp.append(result.id)
        ids.append(temp)
    gt_ids = load_gt_ids()

    for top_k in config.RECALL_CALC_SCOPE:
        recalls, count_all = compare_correct(nq, top_k, rand, gt_ids, ids)
        fname = config.recall_out_fname + '/' + collection_name + '_' + str(search_param) + '_' + str(nq) + "_" + str(
            top_k) + ".csv"
        if not os.path.exists(config.recall_out_fname):
            os.mkdir(config.recall_out_fname)
        with open(fname, 'w') as f:
            f.write('nq,topk,recall\n')
            for i in range(nq):
                line = str(i + 1) + ',' + str(top_k) + ',' + str(recalls[i] * 100) + "%"
                f.write(line + '\n')
            f.write("max, avarage, min\n")
            f.write(str(max(recalls) * 100) + "%," + str(round(count_all / nq / top_k, 3) * 100) + "%," + str(
                min(recalls) * 100) + "%\n")
        print("topk=", top_k, ", total accuracy", round(count_all / nq / top_k, 3) * 100, "%")


def load_gt_ids():
    file_name = config.GROUNDTRUTH_FILE
    gt_ids = []
    result = []
    with open(file_name, 'r') as f:
        for line in f.readlines():
            data = line.split()
            if data:
                result.append(int(data[0]))
            else:
                gt_ids.append(result)
                result = []
    return gt_ids


def compare_correct(nq, top_k, rand, gt_ids, ids):
    recalls = []
    count_all = 0
    for i in range(nq):
        milvus_results = []
        ground_truth = []
        for j in range(top_k):
            milvus_results.append(ids[i][j])
            ground_truth.append(gt_ids[int(rand[i])][j])
            # ground_truth += gt_ids[int(rand[i * top_k]) * config.ground_truth_topk + j]
        # print(milvus_results)
        # print(ground_truth)
        union = list(set(milvus_results).intersection(set(ground_truth)))
        recalls.append(len(union) / top_k)
        count_all += len(union)
    # print("topk_ground_truth:", topk_ground_truth)
    return recalls, count_all

