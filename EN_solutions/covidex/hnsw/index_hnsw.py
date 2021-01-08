import os

import hnswlib
import numpy as np

import helper


class Indexer:
    def __init__(self, folder_path):
        self.folder_path = folder_path

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

    def initialize_hnsw_index(self) -> None:
        # Declaring index
        # possible options are l2, cosine or ip
        self.hnsw = hnswlib.Index(space='l2', dim=self.dim)

        # Initing index - the maximum number of elements should be known beforehand
        # For more configuration, see: https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md
        self.hnsw.init_index(
            max_elements=self.num_elements,
            ef_construction=200,
            M=16)

    def index_and_save(self) -> None:
        print('[HNSW] Starting to index...')
        data = np.empty((0, self.dim))
        data_labels = []
        index_to_uid = []

        for index, uid in enumerate(self.embedding):
            if index % 1000 == 0:
                print(
                    f'[HNSW] Indexed {index}/{self.num_elements}')

            if index % 200 == 0 and len(data_labels) > 0:
                # save progress
                self._add_to_index(data, data_labels,  index)
                # reset
                data = np.empty((0, self.dim))
                data_labels = []

            vector = self.embedding[uid]
            assert len(vector) == self.dim, "Vector dimension mismatch"
            data = np.concatenate((data, [vector]))
            data_labels.append(index)
            index_to_uid.append(uid)

        if len(data_labels) > 0:
            self._add_to_index(data, data_labels, index)
            self._save_index(data, data_labels, index_to_uid, index)

        print('[HNSW] Finished indexing')

    def _add_to_index(self, data, data_labels, index):
        # Element insertion (can be called several times)
        self.hnsw.add_items(data, data_labels)

    def _save_index(self, data, data_labels, index_to_uid, index):
        print('[HNSW] Saving index', index)

        file_name = 'cord19-hnsw'
        output_path = self.get_path(f'{file_name}.bin')
        helper.remove_if_exist(output_path)
        self.hnsw.save_index(output_path)

        # Save index to uid file
        helper.save_index_to_uid_file(
            index_to_uid,
            index,
            self.get_path(f'{file_name}.txt'))


if __name__ == '__main__':
    indexer = Indexer("./api/index/cord19-hnsw-index")
    indexer.load_data()
    indexer.initialize_hnsw_index()
    indexer.index_and_save()
