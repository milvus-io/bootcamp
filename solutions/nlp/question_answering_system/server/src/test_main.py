from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_load():
    _test_upload_file = 'QA_data/example_data.csv'
    _files = {'file': open(_test_upload_file, 'rb')}
    response = client.post("/qa/load_data", files=_files)
    assert response.status_code == 200
    assert response.json() == [{"status": True, "msg": "Successfully loaded data: 99"}, 200]


def test_search():
    response = client.get("/qa/search?question=What insurance should i buy")
    assert response.status_code == 200
    assert response.json()[0]['status'] == True


def test_answer():
    response = client.get("/qa/answer?question=Is  Disability  Insurance  Required  By  Law?")
    assert response.status_code == 200
    assert response.json()['status'] == True


def test_count():
    response = client.post("/qa/count")
    assert response.status_code == 200
    assert response.json() == 99


def test_drop():
    response = client.post("/qa/drop")
    assert response.status_code == 200
