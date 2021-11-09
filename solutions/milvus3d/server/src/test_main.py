from fastapi.testclient import TestClient
import gdown
import os
from main import app


client = TestClient(app)


def download_audio_data():
    # os.system("cd ../data/")
    # data_url = 'https://drive.google.com/uc?id=1nWD8lwlgpA-qOEadkzkxjB1iRCl9nA9u'
    # gdown.download(data_url)
    # os.system("tar -xvf test_load_feature.tar.gz")
    #
    # os.system("mkdir models & cd models")
    # weights_url = "https://drive.google.com/uc?id=1t5jyJ4Ktmlck6GYhNTPVTFZuRP7wPUYq"
    # gdown.download(weights_url)
    # os.system("cd ..")
    # search_data_url = 'https://drive.google.com/uc?id=14fGV3GYcsJR_78XHrxISoceB5bmffkHL'
    # gdown.download(search_data_url)
    # os.system("tar -xvf test_search_data.tar.gz")
    # os.system("rm test_search_data.tar.gz")
    # os.system("rm test_load_feature.tar.gz")
    os.system("chmod +x ../data/test_download_data.sh")
    os.system("../data/test_download_data.sh")


def test_drop():
    response = client.post("/img/drop")
    assert response.status_code == 200


def test_load():
    download_audio_data()
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


# def test_data():
#     response = client.get(
#         "/data?img_file_path=.%2Fexample_audio%2Ftest.wav"
#     )
#     assert response.status_code == 200

