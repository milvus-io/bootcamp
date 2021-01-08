import json
import logging
from datetime import datetime

from fastapi.testclient import TestClient
from freezegun import freeze_time

from app.main import app
from app.models import SearchLogData, SearchLogType

client = TestClient(app)
search_log_data = SearchLogData(query_id='abc', result_id='123', position=0)
fake_date = '2020-01-01'

def log_info_data(type):
    return json.dumps({
        'query_id': search_log_data.query_id,
        'type': type,
        'result_id': search_log_data.result_id,
        'position': search_log_data.position,
        'timestamp': datetime.strptime(fake_date, '%Y-%m-%d').isoformat()})

@freeze_time(fake_date)
def test_log_collapsed(client, mocker):
    mocker.patch('logging.Logger.info')
    response = client.post("/api/search/log/collapsed", json=search_log_data.dict())
    assert response.status_code == 200
    logging.Logger.info.assert_called_once_with(log_info_data(SearchLogType.collapsed))

@freeze_time(fake_date)
def test_log_expanded(client, mocker):
    mocker.patch('logging.Logger.info')
    response = client.post("/api/search/log/expanded", json=search_log_data.dict())
    assert response.status_code == 200
    logging.Logger.info.assert_called_once_with(log_info_data(SearchLogType.expanded))

@freeze_time(fake_date)
def test_log_clicked(client, mocker):
    mocker.patch('logging.Logger.info')
    response = client.post("/api/search/log/clicked", json=search_log_data.dict())
    assert response.status_code == 200
    logging.Logger.info.assert_called_once_with(log_info_data(SearchLogType.clicked))
