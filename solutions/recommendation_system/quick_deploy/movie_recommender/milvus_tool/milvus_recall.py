from pymilvus_orm import *

# from milvus_tool.config import MILVUS_HOST, MILVUS_PORT, top_k, search_param, dim
from milvus_tool.config import *


class RecallByMilvus():
    def __init__(self):
        connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)

    def search(self, vectors, collection_name):
        try:
            if utility.has_collection(collection_name):
                collection = Collection(name = collection_name)
            collection.load()
            res = collection.search(vectors, anns_field="embedding", limit=top_k, param=search_params)
            return res
        except Exception as e:
            print('Milvus recall error: ', e)


if __name__ == '__main__':
    import random
    client = RecallByMilvus()
    collection_name = 'test1'
    partition_name = 'partition_3'
    embeddings = [[random.random() for _ in range(32)] for _ in range(2)]
    res = client.search(collection_name=collection_name, vectors=embeddings)
    for x in res:
        for y in x:
            print(y.id, y.distance)
