import os
from milvus import *



# MILVUS_HOST = '192.168.1.85'
# MILVUS_PORT = 19560

MILVUS_HOST = os.getenv("MILVUS_HOST", "127.0.0.1")
MILVUS_PORT = os.getenv("MILVUS_PORT", 19530)


BERT_HOST = os.getenv("BERT_HOST", "127.0.0.1")
BERT_PORT = os.getenv("BERT_PORT", 5555)


MYSQL_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = os.getenv("MYSQL_PORT", 3306)
MYSQL_USER = os.getenv("MYSQL_USER", 'root')
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", '123456')
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", 'mysql')


TABLE_NAME = os.getenv("DEFAULT_TABLE", "recommend")


insert_file_path = '/data/workspace/lym/Recommend'

batch_size = 10000
temp_file_path = 'temp.csv'


categories_num = 8
texts_num = 20


collection_param = {
    'collection_name': TABLE_NAME,
    'dimension': 768,
    'index_file_size':2048,
    'metric_type':  MetricType.IP
}

search_param = {'nprobe': 16}
top_k = 20

