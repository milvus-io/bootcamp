import os
from pymilvus import *


MILVUS_HOST = 'localhost'
MILVUS_PORT = 19530

dim = 32
pk = FieldSchema(name='pk', dtype=DataType.INT64, is_primary=True)
field = FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, dim=dim)
schema = CollectionSchema(fields=[pk, field], description="movie recommendation: demo films")

index_param = {
    "metric_type": "L2",
    "index_type":"IVF_FLAT",
    "params":{"nlist":128}
    }

top_k = 10
search_params = {
    "metric_type": "L2",
    "params": {"nprobe": 10}
    }
