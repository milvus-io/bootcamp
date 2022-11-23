from fastapi.testclient import TestClient
import gdown
import zipfile
from main import app


client = TestClient(app)


def download_audio_data():
    url = 'https://drive.google.com/uc?id=1bKu21JWBfcZBuEuzFEvPoAX6PmRrgnUp'
    gdown.download(url)

    with zipfile.ZipFile('example_audio.zip', 'r') as zip_ref:
        zip_ref.extractall('./example_audio')

def test_drop():
    response = client.post("/audio/drop")
    assert response.status_code == 200

def test_load():
    download_audio_data()
    response = client.post(
        "/audio/load",
        json={"File": "./example_audio"}
    )
    assert response.status_code == 200
    assert response.json() == {'status': True, 'msg': "Successfully loaded data!"}

def test_progress():
    response = client.get("/progress")
    assert response.status_code == 200
    assert response.json() == "current: 20, total: 20"

def test_count():
    response = client.get("audio/count")
    assert response.status_code == 200
    assert response.json() == 20

def test_search():
    response = client.post(
        "/audio/search/local?query_audio_path=.%2Fexample_audio%2Ftest.wav"
    )
    assert response.status_code == 200
    assert len(response.json()) == 10

def test_data():
    response = client.get(
        "/data?audio_path=.%2Fexample_audio%2Ftest.wav"
    )
    assert response.status_code == 200
