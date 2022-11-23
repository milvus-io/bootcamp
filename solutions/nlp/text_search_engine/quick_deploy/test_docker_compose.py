import requests
# import gdown


def test_load():
    _test_upload_file = '../data/example.csv'
    _files = {'file': open(_test_upload_file, 'rb')}
    response = requests.post("http://127.0.0.1:5000/load", files=_files)
    assert response.status_code == 200
    assert response.json() == "Successfully loaded data!"

def test_count():
    response = requests.post("http://127.0.0.1:5000/count")
    assert response.status_code == 200
    assert response.json() == 160

def test_search():
    response = requests.get(
    "http://127.0.0.1:5000/search?query_sentence=cdFears%20for%20T%20N%20pension%20after%20talks"
    )      
    assert response.status_code == 200

def test_drop():
    response = requests.post("http://127.0.0.1:5000/drop")
    assert response.status_code == 200