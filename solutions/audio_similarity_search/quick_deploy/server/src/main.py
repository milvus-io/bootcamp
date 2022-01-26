import os
from milvus_helpers import MilvusHelper
from mysql_helpers import MySQLHelper
from config import UPLOAD_PATH
from logs import LOGGER
from operations.load import do_load
from operations.search import do_search
from operations.count import do_count
from operations.drop import do_drop
from fastapi import FastAPI
from fastapi import File
from fastapi import UploadFile
from diskcache import Cache
import uvicorn
from starlette.responses import FileResponse
from starlette.requests import Request
from starlette.middleware.cors import CORSMiddleware
from typing import Optional
from pydantic import BaseModel

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

MODEL = None
MILVUS_CLI = MilvusHelper()
MYSQL_CLI = MySQLHelper()

# Mkdir 'tmp/audio-data'
if not os.path.exists(UPLOAD_PATH):
    os.makedirs(UPLOAD_PATH)
    LOGGER.info(f"Mkdir the path: {UPLOAD_PATH}")

@app.get('/data')
def audio_path(audio_path):
    # Get the audio file
    try:
        LOGGER.info(f"Successfully load audio: {audio_path}")
        return FileResponse(audio_path)
    except Exception as e:
        LOGGER.error(f"upload audio error: {e}")
        return {'status': False, 'msg': e}, 400

@app.get('/progress')
def get_progress():
    # Get the progress of dealing with data
    try:
        cache = Cache('./tmp')
        return f"current: {cache['current']}, total: {cache['total']}"
    except Exception as e:
        LOGGER.error(f"Upload data error: {e}")
        return {'status': False, 'msg': e}, 400

class Item(BaseModel):
    Table: Optional[str] = None
    File:str

@app.post('/audio/load')
async def load_audios(item: Item):
    # Insert all the audio files under the file path to Milvus/MySQL
    try:
        total_num = do_load(item.Table, item.File, MILVUS_CLI, MYSQL_CLI)
        LOGGER.info(f"Successfully loaded data, total count: {total_num}")
        return {'status': True, 'msg': "Successfully loaded data!"}
    except Exception as e:
        LOGGER.error(e)
        return {'status': False, 'msg': e}, 400

@app.post('/audio/search')
async def search_audio(request: Request, table_name: str = None, audio: UploadFile = File(...)):
    # Search the uploaded audio in Milvus/MySQL
    try:
        # Save the upload data to server.
        content = await audio.read()
        query_audio_path = os.path.join(UPLOAD_PATH, audio.filename)
        with open(query_audio_path, "wb+") as f:
            f.write(content)
        host = request.headers['host']
        _, paths, distances= do_search(host, table_name, query_audio_path, MILVUS_CLI, MYSQL_CLI)
        names=[]
        for i in paths:
            names.append(os.path.basename(i))
        res = dict(zip(paths, zip(names, distances)))
        # Sort results by distance metric, closest distances first
        res = sorted(res.items(), key=lambda item: item[1][1])
        LOGGER.info("Successfully searched similar audio!")
        return res
    except Exception as e:
        LOGGER.error(e)
        return {'status': False, 'msg': e}, 400

@app.post('/audio/search/local')
async def search_local_audio(request: Request, query_audio_path: str, table_name: str = None):
    # Search the uploaded audio in Milvus/MySQL
    try:
        host = request.headers['host']
        _, paths, distances= do_search(host, table_name, query_audio_path, MILVUS_CLI, MYSQL_CLI)
        names=[]
        for i in paths:
            names.append(os.path.basename(i))
        res = dict(zip(paths, zip(names, distances)))
        # Sort results by distance metric, closest distances first
        res = sorted(res.items(), key=lambda item: item[1][1])
        LOGGER.info("Successfully searched similar audio!")
        return res
    except Exception as e:
        LOGGER.error(e)
        return {'status': False, 'msg': e}, 400

@app.get('/audio/count')
async def count_audio(table_name: str = None):
    # Returns the total number of vectors in the system
    try:
        num = do_count(table_name, MILVUS_CLI)
        LOGGER.info("Successfully count the number of data!")
        return num
    except Exception as e:
        LOGGER.error(e)
        return {'status': False, 'msg': e}, 400

@app.post('/audio/drop')
async def drop_tables(table_name: str = None):
    # Delete the collection of Milvus and MySQL
    try:
        status = do_drop(table_name, MILVUS_CLI, MYSQL_CLI)
        LOGGER.info("Successfully drop tables in Milvus and MySQL!")
        return status
    except Exception as e:
        LOGGER.error(e)
        return {'status': False, 'msg': e}, 400

if __name__ == '__main__':
    uvicorn.run(app=app, host='0.0.0.0', port=8002)
