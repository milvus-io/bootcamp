from milvus_helpers import MilvusHelper
from load import insert_data, create_index
from performance_test import performance, percentile_test
from recall_test import recall
import gdown
import os
import tarfile

client = MilvusHelper()
collection_name = 'pytest'

if not os.path.exists('sift_data'):
    os.mkdir('sift_data')


def test_create_collection():
    assert client.create_collection(collection_name) == True
    assert client.has_collection(collection_name) == True


def test_insert():
    if not os.path.exists('milvus_sift1m.tar.gz'):
        url = "https://drive.google.com/uc?id=1lwS6iAN-ic3s6LLMlUhyxzPlIk7yV7et"
        gdown.download(url)
    file = tarfile.open('milvus_sift1m.tar.gz')
    file.extractall('sift_data')
    insert_data(client, collection_name)
    assert client.count(collection_name) == 1000000


def test_search_performance():
    if not os.path.exists('query_data.tar.gz'):
        url = "https://drive.google.com/uc?id=17jPDk93PQsB5yGh1J1YD9N7X8jvPEUQL"
        gdown.download(url)
    file = tarfile.open('query_data.tar.gz')
    file.extractall('sift_data')
    client.load_data(collection_name)
    load_progress = client.get_loading_progress(collection_name)
    assert load_progress['num_loaded_entities'] == load_progress['num_total_entities']
    performance(client, collection_name, 0)
    result_file = 'performance/test_0_performance.csv'
    assert os.path.exists(result_file)
    assert len(open(result_file, 'r').readlines()) == 13


def test_search_recall():
    if not os.path.exists('gnd.tar.gz'):
        url = "https://drive.google.com/uc?id=1vBP9mKu5oxyognHtOBBRtLvyPvo8cCp0"
        gdown.download(url)
    file = tarfile.open('gnd.tar.gz')
    file.extractall('sift_data')
    assert recall(client, collection_name, 0) == [1, 1, 1, 1]
    client.release_data(collection_name)
    load_progress = client.get_loading_progress(collection_name)
    assert load_progress['num_loaded_entities'] == 0


def test_create_index():
    index_type = "IVF_FLAT"
    create_index(client, collection_name, index_type)
    index_info = client.get_index_params(collection_name)
    assert index_info[0]['index_type'] == 'IVF_FLAT'
    index_progress = client.get_index_progress(collection_name)
    assert index_progress['total_rows'] == index_progress['indexed_rows']


def test_drop_index():
    client.delete_index(collection_name)
    index_info = client.get_index_params(collection_name)
    assert index_info == []


def test_drop_collection():
    client.delete_collection(collection_name)
    assert client.has_collection(collection_name) == False
