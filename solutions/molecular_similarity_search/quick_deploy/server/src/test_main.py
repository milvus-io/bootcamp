from fastapi.testclient import TestClient
from main import app


client = TestClient(app)


def test_load():
    response = client.post(
        "/data/load",
        json={"File": "../../../smiles-data/test_100.smi"}
    )
    assert response.status_code == 200
    assert response.json() == {'status': True, 'msg': "Successfully loaded data!"}

def test_progress():
    response = client.get("/progress")
    assert response.status_code == 200
    assert response.json() == "current: 100, total: 100"

def test_count():
    response = client.post("data/count")
    assert response.status_code == 200
    assert response.json() == 100

def test_search():
    response = client.post(
        "/data/search",
        json={"Mol": "Cc1ccc(cc1)S(=O)(=O)N"}
    )
    assert response.status_code == 200
    assert len(response.json()) == 10

def test_drop():
    response = client.post("/data/drop")
    assert response.status_code == 200