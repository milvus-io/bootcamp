from fastapi.testclient import TestClient
import gdown
import zipfile
from main import app


client = TestClient(app)


def download_video_data():
    url = 'https://drive.google.com/u/0/uc?export=download&confirm=mZQR&id=1CAt-LsF-2gpAMnw5BM75OjCxiR2daulU'
    gdown.download(url)

    with zipfile.ZipFile('examle-gif.zip', 'r') as zip_ref:
        zip_ref.extractall('./examle-gif')

def test_drop():
    response = client.post("/video/drop")
    assert response.status_code == 200

def test_load():
    download_video_data()
    response = client.post(
        "/video/load",
        json={"File": "./examle-gif/100-gif"}
    )
    assert response.status_code == 200
    assert response.json() == {'status': True, 'msg': "Successfully loaded data!"}

def test_progress():
    response = client.get("/progress")
    assert response.status_code == 200
    assert response.json() == {'current': 100, 'total': 100} 

def test_count():
    response = client.post("video/count")
    assert response.status_code == 200
    #assert response.json() == 346

def test_search():
    _test_upload_file = '../../../pic/show.png'
    _files = {'image': open(_test_upload_file, 'rb')}
    response = client.post("/video/search", files=_files)
    # response = client.post(
    #     "/video/search?image=@tumblr_ku4lzkM5fg1qa47qco1_250.gif;type=image/gif"
    # )
    assert response.status_code == 200
    #assert len(response.json()) == 23


def test_data():
    response = client.get(
        "/data?gif_path=.%2Fexamle-gif%2F100-gif%2Ftumblr_ldhw4mUPJB1qcuqc7o1_250.gif"
    )
    assert response.status_code == 200 
