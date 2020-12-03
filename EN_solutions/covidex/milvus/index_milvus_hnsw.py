import os
import numpy as np
import helper
import getopt
import sys
from milvus import Milvus, IndexType, MetricType, Status

class Indexer:
    def __init__(self, folder_path, host, port):
        self.folder_path = folder_path
        self.host = host
        self.port = port
        self.collection_name = 'covdix_collection'
        self.milvus = Milvus(self.host, self.port)


    def get_path(self, path) -> str:
        return os.path.join(self.folder_path, path)


    def load_data(self) -> None:
        self.metadata_path = self.get_path('metadata.csv')
        self.metadata = helper.load_metadata(self.metadata_path)

        self.specter_path = self.get_path('specter.csv')
        self.embedding, self.dim = helper.load_specter_embeddings(
            self.specter_path)
        self.num_elements = len(self.embedding)

        print(f'Number of embeddings: {self.num_elements}')
        print(f'Embedding dimension: {self.dim}')
        assert len(self.metadata) == len(self.embedding), "Data size mismatch"

        status, ok = self.milvus.has_collection(self.collection_name)
        if not ok:
            param = {
                'collection_name': self.collection_name,
                'dimension': self.dim,
                'index_file_size': 1024,  # optional
                'metric_type': MetricType.L2  # optional
            }

            status = self.milvus.create_collection(param)
            print("Milvus create_collection:",status)


    def index_and_save(self) -> None:
        print('[HNSW] Starting to index...')
        data = np.empty((0, self.dim))
        data_labels = []
        index_to_uid = []
        ids = []

        for index, uid in enumerate(self.embedding):
            if index % 10000 == 0:
                print(
                    f'[HNSW] Indexed {index}/{self.num_elements}')

            if index % 200 == 0 and len(data_labels) > 0:
                # save progress
                milvus_ids = self._add_to_index(data, data_labels, index)
                ids += milvus_ids
                # reset
                data = np.empty((0, self.dim))
                data_labels = []

            vector = self.embedding[uid]
            assert len(vector) == self.dim, "Vector dimension mismatch"
            data = np.concatenate((data, [vector]))
            data_labels.append(index)
            index_to_uid.append(uid)

        if len(data_labels) > 0:
            milvus_ids = self._add_to_index(data, data_labels, index)
            ids += milvus_ids
            self._save_index(data, data_labels, index_to_uid, index, ids)

        print('[HNSW] Finished indexing')


    def _add_to_index(self, data, data_labels, index):
        data = data.tolist()
        insert_data = []
        for d in data:
            d_ = [float(i) for i in d]
            insert_data.append(d_)
        # print("--------index--------", len(insert_data),self.collection_name)
        status, ids = self.milvus.insert(collection_name=self.collection_name, records=insert_data)
        # print(status, len(ids))
        return ids


    def _save_index(self, data, data_labels, index_to_uid, index, ids):
        print("-----ids-----", len(ids))

        index_param = {"M": 16, "efConstruction":500}

        print("Creating index: {}".format(index_param))
        status = self.milvus.create_index(self.collection_name, IndexType.HNSW, index_param)
        print('[HNSW] Saving index', status)

        file_name = 'cord19-hnsw-milvus'
        # Save index to uid file
        helper.save_index_to_uid_file(
            index_to_uid,
            ids,
            self.get_path(f'{file_name}.txt'))


if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(
            sys.argv[1:],
            "p:h:",
            ["help", "port=", "host="],
        )
    except getopt.GetoptError:
        print("Usage: test.py --port <milvus_port> --host <milvus_host>")
        sys.exit(2)
    host="127.0.0.1"
    port='19530'
    for opt_name, opt_value in opts:
        if opt_name in ("-h", "--help"):
            print("test.py -q <nq> -k <topk> -t <table> -l -s")
            sys.exit()
        elif opt_name == "--host":
            host = opt_value
        elif opt_name == "--port":
            port = opt_value


    indexer = Indexer("./api/index/cord19-hnsw-index-milvus", host, port)
    indexer.load_data()
    indexer.index_and_save()