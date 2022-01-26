from fastapi.testclient import TestClient
# import gdown
import zipfile
from main import app


client = TestClient(app)



def test_drop():
    response = client.post("/drop")
    assert response.status_code == 200

def test_load():
    _test_upload_file = '../data/example.csv'
    _files = {'file': open(_test_upload_file, 'rb')}
    response = client.post("/load", files=_files)
    assert response.status_code == 200
    assert response.json() == "Successfully loaded data!"


def test_count():
    response = client.post("/count")
    assert response.status_code == 200
    assert response.json() == 160

def test_search():
    response = client.get(
        "/search?query_sentence=cdFears%20for%20T%20N%20pension%20after%20talks"
    )
    assert response.status_code == 200
    assert len(response.json()) == 9
