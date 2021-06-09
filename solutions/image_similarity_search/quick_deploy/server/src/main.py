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
    # Get the image file
    try:
        LOGGER.debug(("Successfully load image: {}".format(image_path))
        return FileResponse(image_path)
    except Exception as e:
        LOGGER.error("upload image error: {}".format(e))
        return {'status': False, 'msg': e}, 400


@app.post('/img/load')
async def load_images(Table: str = None, File: str = None):
    # Insert all the image under the file path to Milvus/MySQL
    try:
        total_num = do_load(Table, File, MODEL, MILVUS_CLI, MYSQL_CLI)
        LOGGER.debug("Successfully insert data, total count: {}".format(total_num))
    except Exception as e:
        LOGGER.error(e)
        return {'status': False, 'msg': e}, 400


@app.post('/img/search')
async def search_images(image: UploadFile = File(...), table_name: str = None):
    # Search the upload image in Milvus/MySQL
    try:
        # Save the upload image to server.
        content = await image.read()
        if not os.path.exists(UPLOAD_PATH):
            os.makedirs(UPLOAD_PATH)
            LOGGER.debug("mkdir the path:{} ".format(UPLOAD_PATH))
        img_path = os.path.join(UPLOAD_PATH, image.filename)
        with open(img_path, "wb+") as f:
            f.write(content)

        paths, distances = do_search(table_name, img_path, MODEL, MILVUS_CLI, MYSQL_CLI)
        res = dict(zip(paths, distances))
        res = sorted(res.items(), key=lambda item: item[1])
        LOGGER.debug("Successfully searched similar images!")
        return res
    except Exception as e:
        LOGGER.error(e)
        return {'status': False, 'msg': e}, 400


@app.post('/img/count')
async def count_images_num(table_name: str = None):
    # Returns the total number of images in the system
    try:
        num = do_count(table_name, MILVUS_CLI)
        LOGGER.debug("Successfully count the number of images!")
        return num
    except Exception as e:
        LOGGER.error(e)
        return {'status': False, 'msg': e}, 400


@app.post('/img/drop')
async def drop_tables(table_name: str = None):
    # Delete the collection of Milvus and MySQL
    try:
        status = do_drop(table_name, MILVUS_CLI, MYSQL_CLI)
        LOGGER.debug("Successfully drop tables in Milvus and MySQL!")
        return status
    except Exception as e:
        LOGGER.error(e)
        return {'status': False, 'msg': e}, 400


if __name__ == '__main__':
    uvicorn.run(app=app, host='127.0.0.1', port=5000)
