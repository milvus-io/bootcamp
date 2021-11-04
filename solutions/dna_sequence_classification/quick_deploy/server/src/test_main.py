from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_drop():
    response = client.post('/text/drop')
    assert response.status_code == 200

def test_count():
    response = client.post('/text/count')
    assert response.status_code == 200

def test_load():
    _test_upload_file = './data/human_data.txt'
    _files = {'file': open(_test_upload_file, 'rb')}
    response = client.post('/text/load', files=_files)
    assert response.status_code == 200

def test_search():
    response = client.get('/text/search?query_sentence=ATGTTCGTGGCATCAGAGAGAAAGATGAGAGCTCACCAGGTGCTCACCTTCCTCCTGCTCTTCGTGATCACCTCGGTGGCCTCTGAAAACGCCAGCACATCCCGAGGCTGTGGGCTGGACCTCCTCCCTCAGTACGTGTCCCTGTGCGACCTGGACGCCATCTGGGGCATTGTGGTGGAGGCGGTGGCCGGGGCGGGCGCCCTGATCACACTGCTCCTGATGCTCATCCTCCTGGTGCGGCTGCCCTTCATCAAGGAGA')
    assert response.status_code == 200
