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



def is_normalized():
    filenames = os.listdir(NL_FOLDER_NAME)
    filenames.sort()
    vetors = load_vec_list(NL_FOLDER_NAME+'/'+filenames[0])
    for i in range(10):
        sqrt_sum = np.sum(np.power(vetors[i], 2))
        print(sqrt_sum)


def connect_server():
    try:
        milvus = Milvus(host=config.MILVUS_HOST, port=config.MILVUS_PORT)
        return milvus
    except Exception as e:
        logging.error(e)


def build_collection(collection_name,it):
    milvus = connect_server()
    if it == 'flat':
        index_type = IndexType.FLAT
        index_param = {'nlist': config.NLIST}
    elif it == 'ivf_flat':
        index_type = IndexType.IVF_FLAT
        index_param = {'nlist': config.NLIST}
    elif it == 'sq8':
        index_param = {'nlist': config.NLIST}
    elif it == 'sq8h':
       # index_type = IndexType.IVF_SQ8H
        index_param = {'nlist': config.NLIST}
    elif it == 'pq':
        index_type = IndexType.IVF_PQ
        index_param = {'nlist': config.NLIST, 'm':config.PQ_M}
    elif it == 'nsg':
        index_type = IndexType.RNSG
        index_param = {'search_length': config.SEARCH_LENGTH, 'out_degree':config.OUT_DEGREE, 'candidate_pool_size':config.CANDIDATE_POOL, 'knng':config.KNNG}
    elif it == 'hnsw':
        index_type = IndexType.HNSW
        index_param = {'M': config.HNSW_M, 'efConstruction':config.EFCONSTRUCTION}
    else:
        print("error index_type, only support these index: flat, ivf_flat, sq8, sq8h, pq, nsg, hnsw")
        print("please try again!")
        sys.exit(2)

   # print(collection_name, " ", index_type, " ", index_param)
    status = milvus.create_index(collection_name, "Vec", {"index_type":"IVF_SQ8", "metric_type": "L2", "params": {"nlist":  config.NLIST}})
    print(status)



def search(collection_name,search_param):
    milvus = connect_server()
    performance_file = config.PERFORMANCE_FILE_NAME
    nq_scope = config.nq_scope
    topk_scope = config.topk_scope
    if not os.path.exists(performance_file):
        os.mkdir(performance_file)
    filename = performance_file + '/' + collection_name + '_' + str(search_param) + '_performance.csv'
    search_params = get_search_params(collection_name,search_param,milvus)
    with open(filename,'w+') as f:
        f.write("nq,topk,total_time,avg_time"+'\n')
        for nq in nq_scope:
            time_start = time.time()
            query_list = load_nq_vec(nq)
            print("load query:", len(query_list), "time_load = ", time.time() - time_start)
            for topk in topk_scope:
                time_start = time.time()
                dsl = {"bool": {"must": [{"vector": {
                    "Vec": {"topk": topk, "query": query_list , "metric_type": "L2", "params": {"nprobe": search_param}}}}]}}
                # results = milvus.search(collection_name, dsl, fields=["Vec"])
                results = milvus.search(collection_name, dsl)
                time_cost = time.time() - time_start
                print(nq, topk, time_cost)
                line = str(nq) + ',' + str(topk) + ',' + str(round(time_cost, 4)) + ',' + str(round(time_cost / nq, 4)) + '\n'
                f.write(line)
            f.write('\n')
    # file.close()
    print("search_vec_list done !")


def get_search_params(collection_name,search_param,milvus):
    info = (milvus.get_collection_info(collection_name))
    index_type= info['fields'][0]['indexes'][0]['index_type']
    if index_type == 'RNSG':
        search_params = {'search_length':search_param}
    elif index_type == 'HNSW':
        search_params = {'ef':search_param}
    else:
        search_params = {'nprobe': search_param}
    return search_params


def load_nq_vec(nq):
    vectors = []
    length = 0
    filenames = os.listdir(config.NQ_FOLDER_NAME)
    filenames.sort()
    for filename in filenames:
        vec_list = load_vec_list(config.NQ_FOLDER_NAME + '/' + filename)
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



def recall_test(collection_name,search_param):
    milvus = connect_server()
    vectors = load_vec_list(config.recall_vec_fname)
    # for nq in config.nq_scope:
    nq = config.recall_nq
    query_list = []
    rand = sorted(random.sample(range(0, len(vectors)), nq))
    for i in rand:
        query_list.append(vectors[i])
    print("load query:", len(query_list))
    #rand=[0,1,2,3,4,5,6,7,8,9]
    search_params = get_search_params(collection_name,search_param,milvus)
    print("collection name:", collection_name, "query list:", len(query_list), "topk:", config.recall_topk, "search_params:", search_params)
    time_start = time.time()
    dsl = {"bool": {"must": [{"vector": {
                    "Vec": {"topk" : 500, "query": query_list, "metric_type": "L2", "params": {"nprobe": search_params}}}}]}}
    results = milvus.search(collection_name,dsl,fields=["Vec"])
    time_cost = time.time() - time_start
    print("time_search = ", time_cost)
    save_re_to_file(collection_name, rand, results, search_param,nq)
    compute_recall(collection_name,nq,results,search_param,rand)




def save_re_to_file(collection_name, rand, results, search_param, nq):
    if not os.path.exists(config.recall_res_fname):
        os.mkdir(config.recall_res_fname)
    file_name = config.recall_res_fname + '/' + collection_name + '_' + str(search_param) + '_' + str(nq) + '_recall.txt'
    entities = results[0]
    with open(file_name, 'w') as f:
        all_ids = entities.ids
        all_distances = entities.distances
        for i in range(len(all_ids)):
                line = str(rand[i]) + ' ' + str(all_ids[i]) + ' ' + str(all_distances[i])
                f.write(line + '\n')
        f.write('\n')
    f.close()

def compute_recall(collection_name,nq,results,search_param,rand):
    all_ids = []
    for i in range (nq):
        temp = []
        entities = results[i]
        temp = entities.ids
       # print(len(temp))
        all_ids.append(temp)
        #print(len(all_ids))
   # entities =results[0]

   # all_ids.append(entities.ids)
   # print(all_ids)
    gt_ids = load_gt_ids() 
    
    for top_k in config.compute_recall_topk:
        recalls, count_all = compare_correct(nq, top_k, rand, gt_ids, all_ids)
        fname = config.recall_out_fname+ '/' + collection_name + '_' + str(search_param) + '_' + str(nq) + "_" + str(top_k) + ".csv"
        with open(fname,'w') as f:
            f.write('nq,topk,recall\n')
           # nq = 1 
            for i in range(nq):#nq
               # top_k = 500
                line = str(i + 1) + ',' + str(top_k) + ',' + str(recalls[i] * 100) + "%"
                f.write(line + '\n')
            f.write("max, avarage, min\n")
            f.write( str(max(recalls) * 100) + "%," + str(round(count_all / nq / top_k, 3) * 100) + "%," + str(min(recalls) * 100) + "%\n")
        print("topk=", top_k,", total accuracy", round(count_all / nq / top_k, 3) * 100, "%")           


def load_gt_ids():
    file_name = config.GT_FNAME_NAME
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



def get_newids(all_ids):
    results=[]
    for o in range(len(all_ids)):
        for p in range(len(all_ids[0])):
           results.append(all_ids[o][p])
          # print(len(results))
    return results

def compare_correct(nq, top_k, rand, gt_ids, all_ids):
    recalls = []
    count_all = 0
   # milvus_results = []
  #  milvus_results=get_newids(all_ids)
   # print(len(milvus_results))
   # top_k=500
    for i in range(nq):
        milvus_results = []
        ground_truth = []
        for j in range(top_k):
            milvus_results.append(all_ids[i][j])
            ground_truth.append(gt_ids[int(rand[i])][j])
        union = list(set(milvus_results).intersection(set(ground_truth)))
        recalls.append(len(union) / top_k)
       # print(recalls)
        count_all += len(union)
       # print(count_all)
    return recalls, count_all
