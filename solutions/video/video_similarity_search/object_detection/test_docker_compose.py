import requests
import gdown
import zipfile

def get_file():
    url = 'https://drive.google.com/uc?id=12AzMujXPw_UjnS63LuwwCOyjkZYVEp3Y'
    gdown.download(url)
    url2 = 'https://drive.google.com/uc?id=17Nh8FULhtJUIZnpZwefKExVzjqlNuUqU'
    gdown.download(url2)
    with zipfile.ZipFile('example_object.zip', 'r') as zip_ref:
        zip_ref.extractall('./data/example_object')
    with zipfile.ZipFile('example.zip', 'r') as zip_ref2:
        zip_ref2.extractall('./data/video')


def test_load():
    get_file()
    response = requests.post(
        "http://127.0.0.1:5000/image/load",
        json={"File": "/data/example_object"}
    )
    assert response.status_code == 200
    assert response.json() == {'status': True, 'msg': "Successfully loaded data!"}

def test_progress():
    response = requests.get("http://127.0.0.1:5000/progress")
    assert response.status_code == 200
    assert response.json() == 'current: 4, total: 4'

def test_count():
    response = requests.post("http://127.0.0.1:5000/image/count")
    assert response.status_code == 200


def test_get_img():
    response = requests.get(
        'http://127.0.0.1:5000/data?image_path=%2Fdata%2Fexample_object%2Fcola.jpg'
        )
    assert response.status_code == 200

def test_search():
    _test_upload_file = './data/video/example_object_video.avi'
    _files = {'video': open(_test_upload_file, 'rb')}
    response = requests.post(
        'http://127.0.0.1:5000/video/search',
        files = _files
    )
    assert response.status_code == 200

def test_drop():
    response = requests.post("http://127.0.0.1:5000/image/drop")
    assert response.status_code == 200
    
