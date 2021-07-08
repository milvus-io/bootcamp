import sys
import os
import time
import numpy as np
import traceback
from bert_serving.client import BertClient
from functools import reduce
import pandas as pd

sys.path.append("..")
from config import DEFAULT_TABLE,TOP_K
from logs import LOGGER




bc = BertClient()




def normaliz_vec(vec_list):
    for i in range(len(vec_list)):
        vec = vec_list[i]
        square_sum = reduce(lambda x,y:x+y, map(lambda x:x*x ,vec))
        sqrt_square_sum = np.sqrt(square_sum)
        coef = 1/sqrt_square_sum
        vec = list(map(lambda x:x*coef, vec))
        vec_list[i] = vec
    return vec_list


# Combine the id of the vector and the question data into a list
def format_data(ids, title_data, text_data):
    data = []
    for i in range(len(ids)):
        value = (str(ids[i]), title_data[i], text_data[i])
        data.append(value)
    return data



# Import vectors to Milvus and data to Mysql respectively
def import_data(collection_name, file_dir,milvus_cli, mysql_cli):
    if not collection_name:
        collection_name = DEFAULT_TABLE
    data = pd.read_csv(file_dir)
    title_data = data['title'].tolist()
    text_data = data['text'].tolist()
    vectors = bc.encode(title_data)
    title_vectors = normaliz_vec(vectors.tolist())
    ids = milvus_cli.insert(collection_name, title_vectors)
    milvus_cli.create_index(collection_name)
    mysql_cli.create_mysql_table(collection_name)
    mysql_cli.load_data_to_mysql(collection_name, format_data(ids, title_data, text_data)) 
    return len(ids)
