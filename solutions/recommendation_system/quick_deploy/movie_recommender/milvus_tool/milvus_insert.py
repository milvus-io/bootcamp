from pymilvus_orm import *

# from milvus_tool.config import MILVUS_HOST, MILVUS_PORT, schema, index_param

from milvus_tool.config import *


class VecToMilvus():
    def __init__(self):
        try:
            connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
            collection = None
        except Exception as e:
            print("Fail to connect Milvus:", e)

    def has_collec(self, collection_name):
        try:
            return utility.has_collection(collection_name)
        except Exception as e:
            print("Milvus has_collec error:", e)

    def set_collection(self, collection_name):
        try:
            if self.has_collec(collection_name):
                self.collection = Collection(name=collection_name)
            else:
                print('No collection {}'.format(collection_name))
        except Exception as e:
                print('Milvus set collection error:', e)

    def creat_collection(self, collection_name):
        try:
            self.collection = Collection(name=collection_name, schema=schema)
            print("Create collection {} successfully".format(collection_name))
            return collection
        except Exception as e:
            print("Milvus create collection error:", e)

    def create_index(self, collection_name):
        try:
            self.set_collection(collection_name)
            status = self.collection.create_index(field_name="embedding", index_params=index_param)
            print("Create index {} successfully".format(collection_name))
            return status
        except Exception as e:
            print("Milvus create index error:", e)

    def has_partition(self, collection_name, partition_name):
        try:
            self.set_collection(collection_name)
            return self.collection.has_partition(partition_name)
        except Exception as e:
            print("Milvus has partition error:", e)

    def create_partition(self, collection_name, partition_name):
        try:
            self.set_collection(collection_name)
            status = self.collection.create_partition(partition_name)
            print('Create partition {} successfully'.format(partition_name))
            return status
        except Exception as e:
            print("Milvus create partition error:", e)

    def drop():
        try:
            self.set_collection(collection_name)
            collection.drop()
            print("Drop collection {}".format(collection_name))
        except Exception as e:
            print("Milvus drop collection error", e)

    def insert(self, ids, vectors, collection_name, partition_name=None):
        try:
            if not self.has_collec(collection_name):
                self.creat_collection(collection_name)
                self.create_index(collection_name)
                print("collection info: {}".format(self.collection))
            if (partition_name is not None) and (not self.has_partition(collection_name, partition_name)):
                self.create_partition(collection_name, partition_name)
            mr = self.collection.insert(data=[ids, vectors], partition_name=partition_name)
            utility.get_connection().flush([collection_name])
            print("Insert {} entities, there are {} entities after insert data.".format(len(ids), self.collection.num_entities))
            return mr
        except Exception as e:
            print("Milvus insert error:", e)


if __name__ == '__main__':
    import random

    client = VecToMilvus()
    collection_name = 'test1'
    partition_name = 'partition_1'
    ids = [random.randint(0, 1000) for _ in range(100)]
    embeddings = [[random.random() for _ in range(dim)] for _ in range(100)]
    mr = client.insert(ids=ids, vectors=embeddings, collection_name=collection_name, partition_name=partition_name)
    print(mr)
    # print(ids)
