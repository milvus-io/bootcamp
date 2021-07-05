import uvicorn
import os
from diskcache import Cache
from fastapi import FastAPI, File, UploadFile
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse
from starlette.requests import Request
from src.helpers.milvus_helpers import MilvusHelper
from src.helpers.mysql_helpers import MySQLHelper
from src.config import UPLOAD_PATH
from src.operations.load import do_load
from src.operations.search import do_search
from src.operations.count import do_count
from src.operations.drop import do_drop
from src.config import TOP_K
from src.logs import LOGGER
from pydantic import BaseModel
from typing import Optional

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

# Mkdir 'tmp/mol-data'
if not os.path.exists(UPLOAD_PATH):
    os.makedirs(UPLOAD_PATH)
    LOGGER.info("mkdir the path:{} ".format(UPLOAD_PATH))


@app.get('/data')
def mols_img(mols_path):
    # Get the molecular image file
    try:
        LOGGER.info(("Successfully load molecular image: {}".format(mols_path)))
        return FileResponse(UPLOAD_PATH + '/' + mols_path + '.png')
    except Exception as e:
        LOGGER.error("upload image error: {}".format(e))
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


class Item(BaseModel):
    Table: Optional[str] = None
    File: str

@app.post('/data/load')
async def load_data(item: Item):
    # Insert all the data under the file path to Milvus/MySQL
    try:
        total_num = do_load(item.Table, item.File, MODEL, MILVUS_CLI, MYSQL_CLI)
        LOGGER.info("Successfully loaded data, total count: {}".format(total_num))
        return {'status': True, 'msg': "Successfully loaded data!"}
    except Exception as e:
        LOGGER.error(e)
        return {'status': False, 'msg': e}, 400


class Item_search(BaseModel):
    Table: Optional[str] = None
    Mol: str
    Num: Optional[int] = TOP_K

@app.post('/data/search')
async def search_data(request: Request, item: Item_search):
    # Search the upload image in Milvus/MySQL
    try:
        # Save the upload data to server.
        ids, paths, distances = do_search(item.Table, item.Mol, item.Num, MODEL, MILVUS_CLI, MYSQL_CLI)
        host = request.headers['host']
        for i in range(len(ids)):
            tmp = "http://" + str(host) + "/data?mols_path=" + str(ids[i])
            ids[i] = tmp
        res = dict(zip(paths, zip(ids, distances)))
        res = sorted(res.items(), key=lambda item: item[1][1])
        LOGGER.info("Successfully searched similar data!")
        return res
    except Exception as e:
        LOGGER.error(e)
        return {'status': False, 'msg': e}, 400


@app.post('/data/count')
async def count_data(table_name: str = None):
    # Returns the total number of data in the system
    try:
        num = do_count(table_name, MILVUS_CLI, MYSQL_CLI)
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
    uvicorn.run(app=app, host='0.0.0.0', port=5000)
