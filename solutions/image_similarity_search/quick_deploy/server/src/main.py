import uvicorn
import os
from fastapi import FastAPI, File, UploadFile
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse
from encode import Resnet50
from milvus_helpers import MilvusHelper
from mysql_helpers import MySQLHelper
from config import UPLOAD_PATH
from operations.load import do_load
from operations.search import do_search
from operations.count import do_count
from operations.drop import do_drop
from logs import write_log

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])
MODEL = Resnet50()
MILVUS_CLI = MilvusHelper()
MYSQL_CLI = MySQLHelper()
LOGGER = write_log()


@app.get('/data')
def image_path(image_path):
    try:
        print("load image:", image_path)
        return FileResponse(image_path)
    except Exception as e:
        write_log(e, 1)
        return {'status': False, 'msg': e}, 400


@app.post('/img/load')
async def load_images(Table: str = None, File: str = None):
    try:
        total_num = do_load(Table, File, MODEL, MILVUS_CLI, MYSQL_CLI)
        LOGGER.info("Load finished, total count: %d" % total_num)
    except Exception as e:
        LOGGER.error(e)
        return {'status': False, 'msg': e}, 400


@app.post('/img/search')
async def search_images(image: UploadFile = File(...), table_name: str = None):
    try:
        # Save the upload image to server.
        content = await image.read()
        print(os.path.exists(UPLOAD_PATH))
        if not os.path.exists(UPLOAD_PATH):
            os.makedirs(UPLOAD_PATH)
        img_path = os.path.join(UPLOAD_PATH, image.filename)
        with open(img_path, "wb+") as f:
            f.write(content)
        # Search similar images and return results.
        paths, distances = do_search(table_name, img_path, MODEL, MILVUS_CLI, MYSQL_CLI)
        res = dict(zip(paths, distances))
        res = sorted(res.items(), key=lambda item: item[1])
        LOGGER.info("Successfully searched similar images!")
        return res
    except Exception as e:
        LOGGER.error(e)
        return {'status': False, 'msg': e}, 400


@app.post('/img/count')
async def count_images_num(table_name: str = None):
    try:
        num = do_count(table_name, MILVUS_CLI)
        LOGGER.info("Successfully count the number of images!")
        return num
    except Exception as e:
        LOGGER.error(e)
        return {'status': False, 'msg': e}, 400


@app.post('/img/drop')
async def drop_tables(table_name: str = None):
    try:
        status = do_drop(table_name, MILVUS_CLI, MYSQL_CLI)
        LOGGER.info("Successfully drop tables in Milvus and MySQL!")
        return status
    except Exception as e:
        LOGGER.error(e)
        return {'status': False, 'msg': e}, 400


if __name__ == '__main__':
    uvicorn.run(app=app, host='127.0.0.1', port=8002)
