import uvicorn
import os
from diskcache import Cache
from fastapi import FastAPI, File, UploadFile
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse
from src.helpers.milvus_helpers import MilvusHelper
from src.helpers.mysql_helpers import MySQLHelper
from src.config import UPLOAD_PATH
from src.operations.load import do_load
from src.operations.search import do_search
from src.operations.count import do_count
from src.operations.drop import do_drop
from src.logs import LOGGER

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])
MODEL = None
MILVUS_CLI = MilvusHelper()
MYSQL_CLI = MySQLHelper()

# Mkdir '/tmp/search-images'
if not os.path.exists(UPLOAD_PATH):
    os.makedirs(UPLOAD_PATH)
    LOGGER.info("mkdir the path:{} ".format(UPLOAD_PATH))


@app.get('/data')
def data_path(data_path):
    # Get the image file
    try:
        LOGGER.info(("Successfully load data: {}".format(data_path)))
        return FileResponse(data_path)
    except Exception as e:
        LOGGER.error("upload data error: {}".format(e))
        return {'status': False, 'msg': e}, 400


@app.get('/progress')
def get_progress():
    # Get the progress of dealing with data
    try:
        cache = Cache('./tmp')
        return "current: {}, total: {}".format(cache['current'], cache['total'])
    except Exception as e:
        LOGGER.error("upload data error: {}".format(e))
        return {'status': False, 'msg': e}, 400


@app.post('/data/load')
async def load_data(Table: str = None, File: str = None):
    # Insert all the data under the file path to Milvus/MySQL
    try:
        total_num = do_load(Table, File, MODEL, MILVUS_CLI, MYSQL_CLI)
        LOGGER.info("Successfully loaded data, total count: {}".format(total_num))
        return {'status': True, 'msg': "Successfully loaded data!"}
    except Exception as e:
        LOGGER.error(e)
        return {'status': False, 'msg': e}, 400


@app.post('/data/search')
async def search_data(Table: str = None, File: str = None):
    # Search the upload image in Milvus/MySQL
    try:
        # Save the upload data to server.
        paths, distances = do_search(Table, File, MODEL, MILVUS_CLI, MYSQL_CLI)
        res = dict(zip(paths, distances))
        res = sorted(res.items(), key=lambda item: item[1])
        LOGGER.info("Successfully searched similar data!")
        return res
    except Exception as e:
        LOGGER.error(e)
        return {'status': False, 'msg': e}, 400


@app.post('/data/count')
async def count_data(table_name: str = None):
    # Returns the total number of data in the system
    try:
        num = do_count(table_name, MILVUS_CLI)
        LOGGER.info("Successfully count the number of data!")
        return num
    except Exception as e:
        LOGGER.error(e)
        return {'status': False, 'msg': e}, 400


@app.post('/data/drop')
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
    uvicorn.run(app=app, host='127.0.0.1', port=5000)