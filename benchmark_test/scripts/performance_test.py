import os
# import config
import time
import numpy as np
from logs import LOGGER
from config import QUERY_FILE_PATH, PERFORMANCE_RESULTS_PATH, NQ_SCOPE, TOPK_SCOPE, METRIC_TYPE


def get_search_params(search_param, index_type):
    if index_type == 'FLAT':
        search_params = {"metric_type": METRIC_TYPE}
    elif index_type == 'RNSG':
        search_params = {"metric_type": METRIC_TYPE, "params": {'search_length': search_param}}
    elif index_type == 'HNSW':
        search_params = {"metric_type": METRIC_TYPE, "params": {'ef': search_param}}
    elif index_type == 'ANNOY':
        search_params = {"metric_type": METRIC_TYPE, "params": {"search_k": search_param}}
    else:
        search_params = {"metric_type": METRIC_TYPE, "params": {"nprobe": search_param}}
        # search_params = {'nprobe': search_param}
    print(search_params)
    return search_params


def get_nq_vec(nq):
    data = np.load(QUERY_FILE_PATH)
    if len(data) > nq:
        return data[0:nq].tolist()
    else:
        LOGGER.info('There is only {} vectors'.format(len(data)))
        return data.tolist()


def performance(client, collection_name, search_param):
    index_type = client.get_index_params(collection_name)
    if index_type:
        index_type = index_type[0]['index_type']
    else:
        index_type = 'FLAT'
    search_params = get_search_params(search_param, index_type)
    if not os.path.exists(PERFORMANCE_RESULTS_PATH):
        os.mkdir(PERFORMANCE_RESULTS_PATH)
    result_filename = collection_name + '_' + str(search_param) + '_performance.csv'
    performance_file = os.path.join(PERFORMANCE_RESULTS_PATH, result_filename)

    with open(performance_file, 'w+') as f:
        f.write("nq,topk,total_time,avg_time" + '\n')
        for nq in NQ_SCOPE:
            query_list = get_nq_vec(nq)
            LOGGER.info("begin to search, nq = {}".format(len(query_list)))
            for topk in TOPK_SCOPE:
                time_start = time.time()
                client.search_vectors(collection_name, query_list, topk, search_params)
                time_cost = time.time() - time_start
                print(nq, topk, time_cost)
                line = str(nq) + ',' + str(topk) + ',' + str(round(time_cost, 4)) + ',' + str(
                    round(time_cost / nq, 4)) + '\n'
                f.write(line)
            f.write('\n')
    LOGGER.info("search_vec_list done !")
