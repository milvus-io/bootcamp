import sys
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from config import MILVUS_HOST, MILVUS_PORT, VECTOR_DIMENSION, METRIC_TYPE
from logs import LOGGER


class MilvusHelper:
    """
    Say something about the ExampleCalass...

    Args:
        args_0 (`type`):
        ...
    """
    def __init__(self):
        try:
            self.collection =None
            connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
            LOGGER.debug(f"Successfully connect to Milvus with IP:{MILVUS_HOST} and PORT:{MILVUS_PORT}")
        except Exception as e:
            LOGGER.error(f"Failed to connect Milvus: {e}")
            sys.exit(1)

    def set_collection(self, collection_name):
        try:
            if self.has_collection(collection_name):
                self.collection = Collection(name=collection_name)
            else:
                raise Exception(f"There is no collection named:{collection_name}")
        except Exception as e:
            LOGGER.error(f"Failed to load data to Milvus: {e}")
            sys.exit(1)

    def has_collection(self, collection_name):
        # Return if Milvus has the collection
        try:
            return utility.has_collection(collection_name)
        except Exception as e:
            LOGGER.error(f"Failed to load data to Milvus: {e}")
            sys.exit(1)

    def create_collection(self, collection_name):
        # Create milvus collection if not exists
        try:
            if not self.has_collection(collection_name):
                field1 = FieldSchema(name="id", dtype=DataType.INT64, descrition="int64", is_primary=True,auto_id=True)
                field2 = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, descrition="float vector", dim=VECTOR_DIMENSION, is_primary=False)
                schema = CollectionSchema(fields=[ field1,field2], description="collection description")
                self.collection = Collection(name=collection_name, schema=schema)
                LOGGER.debug(f"Create Milvus collection: {self.collectio}")
            else:
                self.set_collection(collection_name)
            return "OK"
        except Exception as e:
            LOGGER.error(f"Failed to load data to Milvus: {e}")
            sys.exit(1)

    def insert(self, collection_name, vectors):
        # Batch insert vectors to milvus collection
        try:
            self.create_collection(collection_name)
            data = [vectors]
            self.set_collection(collection_name)
            mr = self.collection.insert(data)
            ids = mr.primary_keys
            self.collection.load()
            LOGGER.debug(
                    f"Insert vectors to Milvus in collection: {collection_name} with {len(vectors)} rows")
            return ids
        except Exception as e:
            LOGGER.error(f"Failed to load data to Milvus: {e}")
            sys.exit(1)

    def create_index(self, collection_name):
        # Create IVF_FLAT index on milvus collection
        try:
            self.set_collection(collection_name)
            default_index= {"index_type": "IVF_SQ8", "metric_type": METRIC_TYPE, "params": {"nlist": 16384}}
            status= self.collection.create_index(field_name="embedding", index_params=default_index)
            if not status.code:
                LOGGER.debug(
                    f"Successfully create index in collection:{collection_name} with param:{default_index}")
                return status
            else:
                raise Exception(status.message)
        except Exception as e:
            LOGGER.error(f"Failed to create index: {e}")
            sys.exit(1)

    def delete_collection(self, collection_name):
        # Delete Milvus collection
        try:
            self.set_collection(collection_name)
            self.collection.drop()
            LOGGER.debug("Successfully drop collection!")
            return "ok"
        except Exception as e:
            LOGGER.error(f"Failed to drop collection: {e}")
            sys.exit(1)

    def search_vectors(self, collection_name, vectors, top_k):
        # Search vector in milvus collection
        try:
            self.set_collection(collection_name)
            search_params = {"metric_type":  METRIC_TYPE, "params": {"nprobe": 16}}
           # data = [vectors]
            res=self.collection.search(vectors, anns_field="embedding", param=search_params, limit=top_k)
            print(res[0])
            LOGGER.debug(f"Successfully search in collection: {res}")
            return res
        except Exception as e:
            LOGGER.error(f"Failed to search vectors in Milvus: {e}")
            sys.exit(1)

    def count(self, collection_name):
        # Get the number of milvus collection
        try:
            self.set_collection(collection_name)
            num =self.collection.num_entities
            LOGGER.debug(f"Successfully get the num:{num} of the collection:{collection_name}")
            return num
        except Exception as e:
            LOGGER.error(f"Failed to count vectors in Milvus: {e}")
            sys.exit(1)

    def get_index_params(self, collection_name):
        # get index info
        self.set_collection(collection_name)
        return [index.params for index in self.collection.indexes]

    def create_partition(self, collection_name, partition_name):
        # create a partition for Milvus
        self.set_collection(collection_name)
        if self.collection.has_partition(partition_name):
            return f"This partition {partition_name} exists"
        else:
            partition = self.collection.create_partition(partition_name)
            return partition

    def delete_index(self, collection_name):
        # drop index
        self.set_collection(collection_name)
        self.collection.drop_index()

    def load_data(self, collection_name):
        # load data from disk to memory
        self.set_collection(collection_name)
        self.collection.load()

    def list_collection(self):
        # List all collections.
        return utility.list_collections()

    def get_loading_progress(self, collection_name):
        # Query the progress of loading.
        return utility.loading_progress(collection_name)

    def get_index_progress(self, collection_name):
        # Query the progress of index building.
        return utility.index_building_progress(collection_name)

    def release_data(self, collection_name):
        # release collection data from memory
        self.set_collection(collection_name)
        return self.collection.release()

    def calculate_distance(self, vectors_left, vectors_right):
        # Calculate distance between two vector arrays.
        return utility.calc_distance(vectors_left, vectors_right)
