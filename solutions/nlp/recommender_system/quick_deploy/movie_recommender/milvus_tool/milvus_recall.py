from pymilvus import connections, utility, Collection
from milvus_tool.config import MILVUS_HOST, MILVUS_PORT, search_params, TOP_K


class RecallByMilvus():
    """
    Search in Milvus
    """
    def __init__(self):
        connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)

    def search(self, vectors, collection_name):
        try:
            if utility.has_collection(collection_name):
                collection = Collection(name = collection_name)
            collection.load()
            res = collection.search(vectors, anns_field="embedding", limit=TOP_K, param=search_params)
            return res
        except Exception as e:
            print('Milvus recall error: ', e)


if __name__ == '__main__':
    import random
    client = RecallByMilvus()
    COLLECTION_NAME = 'test1'
    PARTITION_NAME = 'partition_3'
    embeddings = [[random.random() for _ in range(32)] for _ in range(2)]
    test_res = client.search(collection_name=COLLECTION_NAME, vectors=embeddings)
    for x in test_res:
        for y in x:
            print(y.id, y.distance)
