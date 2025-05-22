from fastapi.testclient import TestClient
import gdown
import zipfile
from main import app

client = TestClient(app)

def get_file():
    url = 'https://drive.google.com/uc?id=1omhIvzbXM9t0mU3hDFqLDpbKpwqyW__b'
    gdown.download(url)

    with zipfile.ZipFile('example_img.zip', 'r') as zip_ref:
        zip_ref.extractall('./')

def test_drop():
    response = client.post('/img/drop')
    assert response.status_code == 200

def test_load_img():
    get_file()
    response = client.post(
    '/img/load',
    json={"File": "./example_img"}
    )
    assert response.status_code == 200

def test_progress():
    response = client.get('/progress')
    assert response.status_code == 200
    assert response.json() == "current: 20, total: 20"

def test_count():
    response = client.post('/img/count')
    assert response.status_code == 200

def test_get_img():
    response = client.get('/data?image_path=.%2Fexample_img%2Ftest.jpg')
    assert response.status_code == 200

def test_upload_img():
    _test_upload_file = './example_img/test.jpg'
    _files = {'image': open(_test_upload_file, 'rb')}
    response = client.post('/img/upload', files = _files)
    assert response.status_code == 200

def test_search():
    _test_upload_file = './example_img/test.jpg'
    _files = {'image': open(_test_upload_file, 'rb')}
    response = client.post('/img/search', files = _files)
    assert response.status_code == 200
