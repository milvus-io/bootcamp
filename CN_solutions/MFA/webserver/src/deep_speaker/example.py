import numpy as np
import random
import os
import sys
from audio import read_mfcc
from batcher import sample_from_mfcc
from constants import SAMPLE_RATE, NUM_FRAMES
from conv_models import DeepSpeakerModel
from test import batch_cosine_similarity
import time
from milvus import Milvus, IndexType, MetricType, Status
from milvus.client.abstract import TopKQueryResult

np.random.seed(123)
random.seed(123)

file_path = 'samples/PhilippeRemy'
model = DeepSpeakerModel()
model.m.load_weights('checkpoints/ResCNN_triplet_training_checkpoint_265.h5', by_name=True)


_HOST = '192.168.1.85'
_PORT = '19530'  # default value
_DIM = 512  # dimension of vector
_INDEX_FILE_SIZE = 32  # max file size of stored index

collection_name = 'example_speaker'

milvus = Milvus()


def voc_to_vec(file):
    mfcc = sample_from_mfcc(read_mfcc(file, SAMPLE_RATE), NUM_FRAMES)
    predict = model.m.predict(np.expand_dims(mfcc, axis=0))
    vec = list(map(float,predict.tolist()[0]))
    return vec


def load_voc(file_path):
    filenames = os.listdir(file_path)
    filenames.sort()
    vectors = []
    ids = []
    for filename in filenames:
        vectors.append(voc_to_vec(file_path + '/' + filename))
        ids.append(int(filename[0:4]))
    return vectors,ids


def connect_milvus_server():
    # print("connect to milvus")
    status = milvus.connect(host=_HOST, port=_PORT, timeout=1000 * 1000 * 20)
    print(status)
    return status


def create_milvus_collection():
    status, ok = milvus.has_collection(collection_name)
    print(status,ok)
    if not ok:
        param = {
            'collection_name': collection_name,
            'dimension': _DIM,
            'index_file_size': _INDEX_FILE_SIZE,  # optional
            'metric_type': MetricType.IP  # optional
        }

        milvus.create_collection(param)

def search_in_milvus(query_vectors):
    param = {
        'collection_name': collection_name,
        'query_records': query_vectors,
        'top_k': 5,
        'params': {"nprobe": 16},
    }
    status, results = milvus.search(**param)
    for re in results:
        print('\n')
        for i in re:
            print(i)

def insert_vec(vectors, ids):
    create_milvus_collection()
    # Insert vectors into demo_collection, return status and vectors id list
    status, ids = milvus.insert(collection_name=collection_name, records=vectors, ids=ids)
    print(status,ids)
    milvus.flush([collection_name])


def main():
    connect_milvus_server()
    vectors, ids = load_voc(file_path)
    insert_vec(vectors, ids)
    query_vectors=[]
    query_vectors.append(vectors[0])
    query_vectors.append(vectors[4])
    query_vectors.append(vectors[8])
    search_in_milvus(vectors)


if __name__ == "__main__":
    main()
