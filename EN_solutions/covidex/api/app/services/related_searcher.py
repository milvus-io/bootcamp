import csv
from typing import Dict, Set

from milvus import Milvus, IndexType, MetricType, Status

from app.settings import settings


class RelatedSearcher:
    def __init__(self):
        self.dim = None
        self.hnsw = None
        self.num_elements: int = None

        self.embedding: Dict = {}
        self.index_to_uid: Dict[int, str] = {}
        self.uid_set: Set[str] = set()

        self.host = settings.host
        self.port = settings.port
        self.collection_name = settings.collection_name
        self.milvus = Milvus(self.host, self.port)

        # Load index files
        self.load_index_to_uid()
        self.load_specter_embedding()


    def load_index_to_uid(self):
        with open(settings.related_milvus_index_to_uid_path, 'r') as f:
            for line in f:
                parsed_line = line.strip().split(' ')
                i, uid = parsed_line
                self.index_to_uid[int(i)] = uid
                self.uid_set.add(uid)
        # print(self.index_to_uid)

        self.num_elements = len(self.index_to_uid)
        print(f'[RelatedSearcher] Loaded {self.num_elements} elements')

    def load_specter_embedding(self):
        res = {}
        dim = None
        print('[RelatedSearcher] Loading SPECTER embeddings')
        with open(settings.related_specter_csv_path, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                uid = row[0]
                vector = row[1:]
                res[uid] = vector

                if dim is None:
                    dim = len(vector)
                else:
                    assert dim == len(
                        vector), "[RelatedSearcher] Embedding dimension mismatch"

        self.embedding = res
        self.dim = dim