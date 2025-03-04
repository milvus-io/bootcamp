"""
Hybrid Semantic Search with Milvus

This demo showcases hybrid semantic search using both dense and sparse vectors with Milvus.
You can optionally use the BGE-M3 model to embed text into dense and sparse vectors, or use randomly generated vectors as an example.
Additionally, you can rerank the search results using the BGE CrossEncoder model.

Prerequisites:
- Milvus 2.4.0 or higher (sparse vector search is available only in these versions).
  Follow this guide to set up Milvus: https://milvus.io/docs/install_standalone-docker.md
- pymilvus Python client library to connect to the Milvus server.
- Optional `model` module in pymilvus for BGE-M3 model.

Installation:
Run the following commands to install the required libraries:
  pip install pymilvus
  pip install pymilvus[model]

Steps:
1. Embed the text as dense and sparse vectors.
2. Set up a Milvus collection to store the dense and sparse vectors.
3. Insert the data into Milvus.
4. Search and inspect the results.
"""

use_bge_m3 = True
use_reranker = True

import random
import numpy as np
import pandas as pd

from pymilvus import (
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    connections,
)

# 1. prepare a small corpus to search
file_path = "quora_duplicate_questions.tsv"
df = pd.read_csv(file_path, sep="\t")
questions = set()
for _, row in df.iterrows():
    obj = row.to_dict()
    questions.add(obj["question1"][:512])
    questions.add(obj["question2"][:512])
    if len(questions) > 10000:
        break

docs = list(questions)

# add some randomly generated texts


def random_embedding(texts):
    rng = np.random.default_rng()
    return {
        "dense": np.random.rand(len(texts), 768),
        "sparse": [
            {
                d: rng.random()
                for d in random.sample(range(1000), random.randint(20, 30))
            }
            for _ in texts
        ],
    }


dense_dim = 768
ef = random_embedding

# BGE-M3 model can embed texts as dense and sparse vectors.
# It is included in the optional `model` module in pymilvus, to install it,
# simply run "pip install pymilvus[model]".
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

ef = BGEM3EmbeddingFunction(use_fp16=False, device="cuda")
dense_dim = ef.dim["dense"]

docs_embeddings = ef(docs)

# 2. setup Milvus collection and index
connections.connect("default", uri="milvus.db")

# Specify the data schema for the new Collection.
fields = [
    # Use auto generated id as primary key
    FieldSchema(
        name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100
    ),
    # Store the original text to retrieve based on semantically distance
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=512),
    # Milvus now supports both sparse and dense vectors,
    # we can store each in a separate field to conduct hybrid search on both vectors
    FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
    FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=dense_dim),
]
schema = CollectionSchema(fields, "")
col_name = "hybrid_demo"
# Now we can create the new collection with above name and schema.
col = Collection(col_name, schema, consistency_level="Strong")

# We need to create indices for the vector fields. The indices will be loaded
# into memory for efficient search.
sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
col.create_index("sparse_vector", sparse_index)
dense_index = {"index_type": "FLAT", "metric_type": "IP"}
col.create_index("dense_vector", dense_index)
col.load()

# 3. insert text and sparse/dense vector representations into the collection
entities = [docs, docs_embeddings["sparse"], docs_embeddings["dense"]]
for i in range(0, len(docs), 50):
    batched_entities = [
        docs[i : i + 50],
        docs_embeddings["sparse"][i : i + 50],
        docs_embeddings["dense"][i : i + 50],
    ]
    col.insert(batched_entities)
col.flush()
