import requests
import gdown
import zipfile

def test_drop():
    response = requests.post("http://127.0.0.1:5000/video/drop")
    assert response.status_code == 200


def get_file():
    url = 'https://drive.google.com/uc?id=1B_euXVJtEAO21HggbzxqhujUd_Q9dN71'
    gdown.download(url)

    with zipfile.ZipFile('examle-gif.zip', 'r') as zip_ref:
        zip_ref.extractall('./data')

def test_load():
    get_file()
    response = requests.post(
        "http://127.0.0.1:5000/video/load",
        json={"File": "/data/100-gif"}
    )
    assert response.status_code == 200
    assert response.json() == {'status': True, 'msg': "Successfully loaded data!"}

def test_progress():
    response = requests.get("http://127.0.0.1:5000/progress")
    assert response.status_code == 200
    assert response.json() == {'current': 10, 'total': 10 } 

def test_count():
    response = requests.post("http://127.0.0.1:5000/video/count")
    assert response.status_code == 200


def test_get_img():
    response = requests.get(
        'http://127.0.0.1:5000/data?gif_path=%2Fdata%2F100-gif%2Ftumblr_ku4lzkM5fg1qa47qco1_250.gif'
        )
    assert response.status_code == 200

def test_search():
    _test_upload_file = '../pic/show.png'
    _files = {'image': open(_test_upload_file, 'rb')}
    response = requests.post(
        'http://127.0.0.1:5000/video/search',
        files = _files
    )
    assert response.status_code == 200



