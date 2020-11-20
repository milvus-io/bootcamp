import logging as log
from common.config import MILVUS_TABLE, OUT_PATH, OUT_DATA
from indexer.index import milvus_client, search_vectors, get_vector_by_ids
from indexer.tools import connect_mysql, search_by_milvus_id
import numpy as np
import torch
import pickle
import dgl
import json
import random




def do_search(index_client, conn, cursor, Text, Image):





  #_, vector_item = get_vector_by_ids(index_client, Text, Image)



    status, results = search_vectors(index_client, Text, vector_item)
    print("-----milvus search status------", status)

    results_ids = []
    for results_id in results.id_array:
        for i in results_id:
            img = str(i) +'.jpg'
            if img in img_list and i not in search_id:
                results_ids.append(img)
    # print(results_ids)
    try:
        list_ids = random.sample(results_ids, 10)
    except:
        list_ids = results_ids
    return list_ids