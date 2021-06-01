import time
from milvus import Milvus, IndexType, MetricType, Status
from common.config import DEFAULT_MILVUS_TABLE_NAME


class MilvusOperator:
    def __init__(self, addr, port):
        self.addr = addr
        self.port = port
        self.client = self.milvus_client()

    def __exit__(self):
        self.client.disconnect()

    def milvus_client(self):
        try:
            milvus = Milvus(host=self.addr, port=self.port)
            return milvus
        except Exception as e:
            print(e)

        

    def check_before_use(self):
        status, exists = self.client.has_collection(DEFAULT_MILVUS_TABLE_NAME)
        if not exists:
            self.create_table()
            time.sleep(3)

    def create_table(self):
        param = {
            'collection_name': 'video_search',
            'dimension': 512,
            'index_file_size': 1024,
            'metric_type': MetricType.L2
        }
        status = self.client.create_collection(param)

    def insert_feats(self, feats):
        try:
            self.check_before_use()
            status, ids = self.client.insert(collection_name=DEFAULT_MILVUS_TABLE_NAME,
                                                  records=feats)
            return ids
        except Exception as e:
            print(e)

    def search(self, feats, num):
        search_param = {'nprobe': 16}
        status, results = self.client.search(
            collection_name=DEFAULT_MILVUS_TABLE_NAME,
            query_records=feats,
            top_k=num,
            params=search_param)
        return results
