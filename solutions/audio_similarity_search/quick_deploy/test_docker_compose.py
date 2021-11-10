import requests
import gdown
import zipfile


def download_audio_data():
    url = 'https://drive.google.com/uc?id=1bKu21JWBfcZBuEuzFEvPoAX6PmRrgnUp'
    gdown.download(url)

    with zipfile.ZipFile('example_audio.zip', 'r') as zip_ref:
        zip_ref.extractall('./data')

def test_drop():
    response = requests.post("http://127.0.0.1:8002/audio/drop")
    assert response.status_code == 200

def test_load():
    download_audio_data()
    response = requests.post(
        "http://127.0.0.1:8002/audio/load",
        json={"File": "/audio_data"}
    )
    assert response.status_code == 200
    assert response.json() == {'status': True, 'msg': "Successfully loaded data!"}

def test_progress():
    response = requests.get("http://127.0.0.1:8002/progress")
    assert response.status_code == 200
    assert response.json() == "current: 20, total: 20"

def test_count():
    response = requests.get("http://127.0.0.1:8002/audio/count")
    assert response.status_code == 200
    assert response.json() == 20

def test_search():
    response = requests.post(
        "http://127.0.0.1:8002/audio/search/local?query_audio_path=%2Faudio_data%2Ftest.wav"
    )
    assert response.status_code == 200
    assert len(response.json()) == 10

def test_data():
    response = requests.get(
        "http://127.0.0.1:8002/data?audio_path=%2Faudio_data%2Ftest.wav"
    )
    assert response.status_code == 200
