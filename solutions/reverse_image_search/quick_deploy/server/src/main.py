import uvicorn
import os
from diskcache import Cache
from fastapi import FastAPI, File, UploadFile
from fastapi.param_functions import Form
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse
from encode import Resnet50
from milvus_helpers import MilvusHelper
from mysql_helpers import MySQLHelper
from config import TOP_K, UPLOAD_PATH
from operations.load import do_load
from operations.upload import do_upload
from operations.search import do_search
from operations.count import do_count
from operations.drop import do_drop
from logs import LOGGER
from pydantic import BaseModel
from typing import Optional
from urllib.request import urlretrieve

app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],

)
MODEL = Resnet50()
MILVUS_CLI = MilvusHelper()
MYSQL_CLI = MySQLHelper()

# Mkdir '/tmp/search-images'
if not os.path.exists(UPLOAD_PATH):
    os.makedirs(UPLOAD_PATH)
    LOGGER.info(f"mkdir the path:{UPLOAD_PATH}")

@app.get('/data')
def get_img(image_path):
    # Get the image file
    try:
        LOGGER.info(f"Successfully load image: {image_path}")
        return FileResponse(image_path)
    except Exception as e:
        LOGGER.error(f"Get image error: {e}")
        return {'status': False, 'msg': e}, 400


@app.get('/progress')
def get_progress():
    # Get the progress of dealing with images
    try:
        cache = Cache('./tmp')
        return f"current: {cache['current']}, total: {cache['total']}"
    except Exception as e:
        LOGGER.error(f"upload image error: {e}")
        return {'status': False, 'msg': e}, 400

class Item(BaseModel):
    Table: Optional[str] = None
    File: str

@app.post('/img/load')
async def load_images(item: Item):
    # Insert all the image under the file path to Milvus/MySQL
    try:
        total_num = do_load(item.Table, item.File, MODEL, MILVUS_CLI, MYSQL_CLI)
        LOGGER.info(f"Successfully loaded data, total count: {total_num}")
        return "Successfully loaded data!"
    except Exception as e:
        LOGGER.error(e)
        return {'status': False, 'msg': e}, 400

@app.post('/img/upload')
async def upload_images(image: UploadFile = File(None), url: str = None, table_name: str = None):
    # Insert the upload image to Milvus/MySQL
    try:
        # Save the upload image to server.
        if image is not None:
            content = await image.read()
            print('read pic succ')
            img_path = os.path.join(UPLOAD_PATH, image.filename)
            with open(img_path, "wb+") as f:
                f.write(content)
        elif url is not None:
            img_path = os.path.join(UPLOAD_PATH, os.path.basename(url))
            urlretrieve(url, img_path)
        else:
            return {'status': False, 'msg': 'Image and url are required'}, 400
        vector_id = do_upload(table_name, img_path, MODEL, MILVUS_CLI, MYSQL_CLI)
        LOGGER.info(f"Successfully uploaded data, vector id: {vector_id}")
        return "Successfully loaded data: " + str(vector_id)
    except Exception as e:
        LOGGER.error(e)
        return {'status': False, 'msg': e}, 400

@app.post('/img/search')
async def search_images(image: UploadFile = File(...), topk: int = Form(TOP_K), table_name: str = None):
    # Search the upload image in Milvus/MySQL
    try:
        # Save the upload image to server.
        content = await image.read()
        print('read pic succ')
        img_path = os.path.join(UPLOAD_PATH, image.filename)
        with open(img_path, "wb+") as f:
            f.write(content)
        paths, distances = do_search(table_name, img_path, topk, MODEL, MILVUS_CLI, MYSQL_CLI)
        res = dict(zip(paths, distances))
        res = sorted(res.items(), key=lambda item: item[1])
        LOGGER.info("Successfully searched similar images!")
        return res
    except Exception as e:
        LOGGER.error(e)
        return {'status': False, 'msg': e}, 400


@app.post('/img/count')
async def count_images(table_name: str = None):
    # Returns the total number of images in the system
    try:
        num = do_count(table_name, MILVUS_CLI)
        LOGGER.info("Successfully count the number of images!")
        return num
    except Exception as e:
        LOGGER.error(e)
        return {'status': False, 'msg': e}, 400


@app.post('/img/drop')
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
