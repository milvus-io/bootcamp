import requests
import gdown
import zipfile

def get_file():
    url = 'https://drive.google.com/uc?id=1omhIvzbXM9t0mU3hDFqLDpbKpwqyW__b'
    gdown.download(url)

    with zipfile.ZipFile('example_img.zip', 'r') as zip_ref:
        zip_ref.extractall('./data')

def test_load():
    get_file()
    response = requests.post(
        "http://127.0.0.1:5000/img/load",
        json={"File": "/data/example_img"}
    )
    assert response.status_code == 200
    assert response.json() == "Successfully loaded data!"

def test_progress():
    response = requests.get("http://127.0.0.1:5000/progress")
    assert response.status_code == 200
    assert response.json() == "current: 20, total: 20"

def test_count():
    response = requests.post("http://127.0.0.1:5000/img/count")
    assert response.status_code == 200
    assert response.json() == 20

def test_get_img():
    response = requests.get(
        'http://127.0.0.1:5000/data?image_path=%2Fdata%2Fexample_img%2Ftest.jpg'
        )
    assert response.status_code == 200

def test_upload_img():
    _test_upload_file = './data/example_img/test.jpg'
    _files = {'image': open(_test_upload_file, 'rb')}
    response = requests.post(
        'http://127.0.0.1:5000/img/upload',
        files = _files
        )
    assert response.status_code == 200

def test_search():
    _test_upload_file = './data/example_img/test.jpg'
    _files = {'image': open(_test_upload_file, 'rb')}
    response = requests.post(
        'http://127.0.0.1:5000/img/search',
        files = _files
    )
    assert response.status_code == 200


def test_drop():
    response = requests.post("http://127.0.0.1:5000/img/drop")
    assert response.status_code == 200
