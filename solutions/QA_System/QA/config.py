import os
from milvus import *

MILVUS_HOST = os.getenv("MILVUS_HOST", "127.0.0.1")
MILVUS_PORT = os.getenv("MILVUS_PORT", 19530)


PG_HOST = os.getenv("PG_HOST", "127.0.0.1")
PG_PORT = os.getenv("PG_PORT", 5432)
PG_USER = os.getenv("PG_USER", "postgres")
PG_PASSWORD = os.getenv("PG_PASSWORD", "postgres")
PG_DATABASE = os.getenv("PG_DATABASE", "postgres")

BERT_HOST = os.getenv("BERT_HOST", "127.0.0.1")
BERT_PORT = os.getenv("BERT_PORT", 5555)
# BERT_PORT_EN = os.getenv("BERT_PORT_EN", 6666)
# BERT_PORT_ZH = os.getenv("BERT_PORT_ZH", 5555)

# BERT_PORT_out_EN = os.getenv("BERT_PORT_out_EN", 6667)
# BERT_PORT_out_ZH = os.getenv("BERT_PORT_out_ZH", 5556)

DEFAULT_TABLE = os.getenv("DEFAULT_TABLE", "qa_system_1")

collection_param = {
    'collection_name': DEFAULT_TABLE,
    'dimension': 768,
    'index_file_size':2048,
    'metric_type':  MetricType.IP
}

search_param = {'nprobe': 32}
top_k = 5


# user_table_name = 'user_info_1'
