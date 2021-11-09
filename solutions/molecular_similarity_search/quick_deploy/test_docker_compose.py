import requests


def test_load():
    response = requests.post(
        "http://127.0.0.1:5000/data/load",
        json={"File": "/mols_data/test_100.smi"}
    )
    assert response.status_code == 200
    assert response.json() == {'status': True, 'msg': "Successfully loaded data!"}

def test_progress():
    response = requests.get("http://127.0.0.1:5000/progress")
    assert response.status_code == 200
    assert response.json() == "current: 100, total: 100"

def test_count():
    response = requests.post("http://127.0.0.1:5000/data/count")
    assert response.status_code == 200
    assert response.json() == 100

def test_search():
    response = requests.post(
        "http://127.0.0.1:5000/data/search",
        json={"Mol": "Cc1ccc(cc1)S(=O)(=O)N"}
    )
    assert response.status_code == 200
    assert len(response.json()) == 10

def test_drop():
    response = requests.post("http://127.0.0.1:5000/data/drop")
    assert response.status_code == 200
