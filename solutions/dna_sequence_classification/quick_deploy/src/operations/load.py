import sys
import os
import time
import numpy as np
#import traceback
import pandas as pd
import pickle
from sklearn import preprocessing

sys.path.append("..")
from config import DEFAULT_TABLE,TOP_K, VECTOR_DIMENSION, KMER_K
from utils import *
from logs import LOGGER

# Function to replace sequence column with kmers in df
def seq_to_kmers(df):
    df['kmers'] = df.apply(lambda x: build_kmers(x['sequence'], KMER_K), axis =1)
    df = df.drop(['sequence'],axis=1)

# Get lists of sequences in k-mers and labels in text from dataframe
def get_vectors(df):
    seq_to_kmers(df)
    words = list(df['kmers']) # list of all sequences in kmers
    texts = []
    for i in range(len(words)):
        texts.append(' '.join(words[i]))
    vectors = train_vec(texts)
    return vectors

def format_data(ids, classes):
    data = []
    for i in range(len(ids)):
        value = (str(ids[i]), str(classes[i]))
        data.append(value)
    return data

# Import vectors to Milvus and data to Mysql respectively
def import_data(collection_name, file_dir, milvus_cli, mysql_cli):
    if not collection_name:
        collection_name = DEFAULT_TABLE
    df = pd.read_table(file_dir)
    class_name = collection_name+'_class'
    vectors = get_vectors(df)
    ids = milvus_cli.insert(collection_name, vectors)
    milvus_cli.create_index(collection_name)
    mysql_cli.create_mysql_table(collection_name)
    mysql_cli.load_data_to_mysql(collection_name, format_data(ids, df['class']))
    return len(ids)

"""
from milvus_helpers import MilvusHelper
from mysql_helpers import MySQLHelper
if __name__ == "__main__":
    COLLECTION_NAME = "test"
    file = "/Users/mengjiagu/Desktop/human.txt"
    MILVUS_CLI = MilvusHelper()
    MYSQL_CLI = MySQLHelper()
    import_data(COLLECTION_NAME, file, MILVUS_CLI, MYSQL_CLI)
"""
