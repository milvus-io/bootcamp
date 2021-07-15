import numpy as np
import random
import time
import os
from performance_test import get_search_params
from config import RECALL_QUERY_FILE, RECALL_NQ, METRIC_TYPE, RECALL_TOPK, RECALL_RES, RECALL_CALC_SCOPE, \
    RECALL_RES_TOPK, GROUNDTRUTH_FILE


def save_search_res(collection_name, rand, results, search_param, nq):
    if not os.path.exists(RECALL_RES):
        os.mkdir(RECALL_RES)
    file_name = os.path.join(RECALL_RES, collection_name + '_' + str(search_param) + '_' + str(nq) + '_recall.txt')
    with open(file_name, 'w') as f:
        i = 0
        for result in results:
            for res in result:
                line = str(rand[i]) + ' ' + str(res.id) + ' ' + str(res.distance)
                f.write(line + '\n')
            f.write('\n')
            i = i + 1


def compute_recall(collection_name, nq, results, search_param, rand):
    ids = []
    for result in results:
        temp = []
        for res in result:
            temp.append(res.id)
        ids.append(temp)

    gt_ids = load_gt_ids()

    for top_k in RECALL_CALC_SCOPE:
        recalls, count_all = compare_correct(nq, top_k, rand, gt_ids, ids)

        if not os.path.exists(RECALL_RES_TOPK):
            os.makedirs(RECALL_RES_TOPK)
        fname = collection_name + '_' + str(search_param) + '_' + str(nq) + "_" + str(top_k) + ".csv"
        fname = os.path.join(RECALL_RES_TOPK, fname)
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
    gt_ids = []
    result = []
    with open(GROUNDTRUTH_FILE, 'r') as f:
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


def recall(client, collection_name, search_param):
    data = np.load(RECALL_QUERY_FILE).tolist()
    query_list = []
    rand = sorted(random.sample(range(0, len(data)), RECALL_NQ))
    for i in rand:
        query_list.append(data[i])
    index_type = client.get_index_params(collection_name)
    if index_type:
        index_type = index_type[0]['index_type']
    else:
        index_type = 'FLAT'
    # index_type = index_type[0]['index_type']
    search_params = get_search_params(search_param, index_type)
    print("collection name:", collection_name, "query list:", len(query_list), "topk:", RECALL_TOPK,
          "search_params:", search_params)
    time_start = time.time()
    results = client.search_vectors(collection_name, query_list, RECALL_TOPK, search_params)
    time_cost = time.time() - time_start
    print("time_search = ", time_cost)
    save_search_res(collection_name, rand, results, search_param, RECALL_NQ)
    compute_recall(collection_name, RECALL_NQ, results, search_param, rand)
