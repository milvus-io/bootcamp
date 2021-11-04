from milvus_helpers import MilvusHelper
from load import insert_data, create_index
from performance_test import performance, percentile_test
from config import BASE_FILE_PATH
import gdown
import os
import tarfile

client = MilvusHelper()
collection_name = 'test'


def test_create_collection():
    assert client.create_collection(collection_name) == True
    assert client.has_collection(collection_name) == True

'''
def test_insert():
    if not os.path.exists('milvus_sift1m.tar.gz'):
        url = "https://drive.google.com/uc?id=1lwS6iAN-ic3s6LLMlUhyxzPlIk7yV7et"
        gdown.download(url)
    if not os.path.exists('sift_data'):
        os.mkdir('sift_data')
    file = tarfile.open('milvus_sift1m.tar.gz')
    file.extractall('sift_data')
    insert_data(client, collection_name)
    assert client.count(collection_name) == 1000000
'''

def test_create_index():
    pass

def test_search_performance():
    if not os.path.exists('query_data.tar.gz'):
        url = "https://drive.google.com/uc?id=17jPDk93PQsB5yGh1J1YD9N7X8jvPEUQL"
        gdown.download(url)
    if not os.path.exists('sift_data'):
        os.mkdir('sift_data')
    file = tarfile.open('query_data.tar.gz')
    file.extractall('sift_data')
    assert client.load_data(collection_name) == True
    performance(client, collection_name, 10)
    result_file = 'performance/test_10_performance.csv'
    assert os.path.exists(result_file)
    assert len(open(result_file, 'r').readlines()) == 13

def test_search_recall():
    if not os.path.exists('gnd.tar.gz'):
        url = "https://drive.google.com/uc?id=1vBP9mKu5oxyognHtOBBRtLvyPvo8cCp0"
        gdown.download(url)
    if not os.path.exists('sift_data'):
        os.mkdir('sift_data')
    file = tarfile.open('gnd.tar.gz')
    file.extractall('sift_data')





