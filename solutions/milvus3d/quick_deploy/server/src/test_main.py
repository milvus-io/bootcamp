from fastapi.testclient import TestClient
import os
from main import app


client = TestClient(app)


def test_download_audio_data():

    os.system("chmod +x ../data/test_download_data.sh")
    os.system("../data/test_download_data.sh")


def test_drop():
    response = client.post("/img/drop")
    assert response.status_code == 200


def test_load():
    # download_audio_data()
    response = client.post(
        "/img/load",
        json={"File": "../data/test_load_feature"}
    )
    assert response.status_code == 200
    assert response.json() == "Successfully loaded data, total count: 40"


def test_progress():
    response = client.get("/progress")
    assert response.status_code == 200
    assert response.json() == "current: 40, total: 40"


def test_count():
    response = client.post("/img/count")
    assert response.status_code == 200
    assert response.json() == 40


def test_search():
    response = client.post(
        "/img/search?model_path=..%2Fdata%2Ftest_search_data%2Fdesk_0001.off"
    )
    assert response.status_code == 200
    assert len(response.json()) == 10


