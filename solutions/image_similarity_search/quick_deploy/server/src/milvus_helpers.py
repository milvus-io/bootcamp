import sys
from config import MILVUS_HOST, MILVUS_PORT, VECTOR_DIMENSION, INDEX_FILE_SIZE, METRIC_TYPE
from milvus import Milvus, IndexType
from main import LOGGER


class MilvusHelper:
    def __init__(self):
        try:
            self.client = Milvus(host=MILVUS_HOST, port=MILVUS_PORT)
            LOGGER.debug("Successfully connect to Milvus with IP:{} and PORT:{}".format(MILVUS_HOST, MILVUS_PORT))
        except Exception as e:
            LOGGER.error("Failed to connect Milvus: {}".format(e))
            sys.exit(1)

    # Create milvus collection if not exists
    def create_colllection(self, collection_name):
        try:
            if not self.client.has_collection(collection_name)[1]:
                collection_param = {
                    'collection_name': collection_name,
                    'dimension': VECTOR_DIMENSION,
                    'index_file_size': INDEX_FILE_SIZE,
                    'metric_type': METRIC_TYPE
                }
                status = self.client.create_collection(collection_param)
                if status.code != 0:
                    raise Exception(status.message)
                LOGGER.debug("Create Milvus collection: {}".format(collection_name))
        except Exception as e:
            LOGGER.error("Failed to load data to Milvus: {}".format(e))
            sys.exit(1)

    # Batch insert vectors to milvus collection
    def insert(self, collection_name, vectors):
        try:
            self.create_colllection(collection_name)
            status, ids = self.client.insert(collection_name=collection_name, records=vectors)
            if not status.code:
                LOGGER.debug("Insert vectors to Milvus in collection: {} with {} rows".format(collection_name, len(vectors)))
                return ids
            else:
                raise Exception(status.message)
        except Exception as e:
            LOGGER.error("Failed to load data to Milvus: {}".format(e))
            sys.exit(1)

    # Create IVF_FLAT index on milvus collection
    def create_index(self, collection_name):
        try:
            index_param = {'nlist': 16384}
            status = self.client.create_index(collection_name, IndexType.IVF_FLAT, index_param)
            if not status.code:
                LOGGER.debug("Successfully create index in collection:{} with param:{}".format(collection_name, index_param))
                return status
            else:
                raise Exception(status.message)
        except Exception as e:
            LOGGER.error("Failed to create index: {}".format(e))
            sys.exit(1)

    # Delete Milvus collection
    def delete_collection(self, collection_name):
        try:
            status = self.client.drop_collection(collection_name=collection_name)
            if not status.code:
                LOGGER.debug("Successfully drop collection: {}".format(collection_name))
                return status
            else:
                raise Exception(status.message)
        except Exception as e:
            LOGGER.error("Failed to drop collection: {}".format(e))
            sys.exit(1)

    # Search vector in milvus collection
    def search_vectors(self, collection_name, vectors, top_k):
        try:
            search_param = {'nprobe': 16}
            status, result = self.client.search(collection_name=collection_name, query_records=vectors, top_k=top_k,
                                                params=search_param)
            if not status.code:
                LOGGER.debug("Successfully search in collection: {}".format(collection_name))
                return result
            else:
                raise Exception(status.message)
        except Exception as e:
            LOGGER.error("Failed to search vectors in Milvus: {}".format(e))
            sys.exit(1)

    # Get the number of milvus collection
    def count(self, collection_name):
        try:
            status, num = self.client.count_entities(collection_name=collection_name)
            if not status.code:
                LOGGER.debug("Successfully get the num:{} of the collection:{}".format(num, collection_name))
                return num
            else:
                raise Exception(status.message)
        except Exception as e:
            LOGGER.error("Failed to count vectors in Milvus: {}".format(e))
            sys.exit(1)
