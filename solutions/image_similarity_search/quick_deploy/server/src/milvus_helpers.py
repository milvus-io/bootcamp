import sys
from config import MILVUS_HOST, MILVUS_PORT, VECTOR_DIMENSION, INDEX_FILE_SIZE, METRIC_TYPE
from milvus import Milvus, IndexType
from main import LOGGER


class MilvusHelper:
    def __init__(self):
        try:
            self.client = Milvus(host=MILVUS_HOST, port=MILVUS_PORT)
        except Exception as e:
            LOGGER.error("Failed to connect Milvus: ", e)
            sys.exit(1)

    def insert(self, collection, embeddings):
        try:
            if not self.client.has_collection(collection)[1]:
                collection_param = {
                    'collection_name': collection,
                    'dimension': VECTOR_DIMENSION,
                    'index_file_size': INDEX_FILE_SIZE,
                    'metric_type': METRIC_TYPE
                }
                status = self.client.create_collection(collection_param)
                if status.code != 0:
                    raise Exception(status.message)
            status, ids = self.client.insert(collection_name=collection, records=embeddings)
            if not status.code:
                return ids
            else:
                raise Exception(status.message)
        except Exception as e:
            LOGGER.error("Failed to load data to Milvus: ", e)
            sys.exit(1)

    def create_index(self, collection):
        try:
            param = {'nlist': 16384}
            status = self.client.create_index(collection, IndexType.IVF_FLAT, param)
            if not status.code:
                return status
            else:
                raise Exception(status.message)
        except Exception as e:
            LOGGER.error("Failed to create index: ", e)
            sys.exit(1)

    def delete_collection(self, collection):
        try:
            status = self.client.drop_collection(collection_name=collection)
            if not status.code:
                return status
            else:
                raise Exception(status.message)
        except Exception as e:
            LOGGER.error("Failed to drop collection: ", e)
            sys.exit(1)

    def search_vectors(self, collection, vectors, top_k):
        try:
            search_param = {'nprobe': 16}
            status, result = self.client.search(collection_name=collection, query_records=vectors, top_k=top_k,
                                                params=search_param)
            if not status.code:
                return result
            else:
                raise Exception(status.message)
        except Exception as e:
            LOGGER.error("Failed to search vectors in Milvus: ", e)
            sys.exit(1)

    def count(self, collection):
        try:
            status, num = self.client.count_entities(collection_name=collection)
            if not status.code:
                return num
            else:
                raise Exception(status.message)
        except Exception as e:
            LOGGER.error("Failed to count vectors in Milvus: ", e)
            sys.exit(1)
