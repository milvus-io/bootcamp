from fastapi.testclient import TestClient
import gdown
import zipfile
from main import app


client = TestClient(app)


def download_video_data():
    url = 'https://drive.google.com/uc?id=12AzMujXPw_UjnS63LuwwCOyjkZYVEp3Y'
    gdown.download(url)
    url2 = 'https://drive.google.com/uc?id=17Nh8FULhtJUIZnpZwefKExVzjqlNuUqU'
    gdown.download(url2)
    with zipfile.ZipFile('example_object.zip', 'r') as zip_ref:
        zip_ref.extractall('./data/example_object')
    with zipfile.ZipFile('example.zip', 'r') as zip_ref2:
        zip_ref2.extractall('./data/example_video')

def test_drop():
    response = client.post("/image/drop")
    assert response.status_code == 200

def test_load():
    download_video_data()
    response = client.post(
        "/image/load",
        json={"File": "./data/example_object"}
    )
    assert response.status_code == 200
    assert response.json() == {'status': True, 'msg': "Successfully loaded data!"}

def test_progress():
    response = client.get("/progress")
    assert response.status_code == 200
    assert response.json() == 'current: 4, total: 4'

def test_count():
    response = client.post("image/count")
    assert response.status_code == 200
    assert response.json() == 4

def test_search():
    _test_upload_file = './data/example_video/example_object_video.avi'
    _files = {'video': open(_test_upload_file, 'rb')}
    response = client.post("/video/search", files=_files)
    assert response.status_code == 200

def test_data():
    response = client.get(
        "/data?image_path=.%2Fdata%2Fexample_object%2Fcola.jpg"
    )
    assert response.status_code == 200
