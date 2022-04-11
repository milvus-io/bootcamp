import uvicorn
import os
from fastapi import FastAPI, File, UploadFile
from diskcache import Cache
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
from encode import CustomOperator
from milvus_helpers import MilvusHelper
from mysql_helpers import MySQLHelper
from operations.load import do_load
from operations.search import do_search
from operations.count import do_count
from operations.drop import do_drop
from logs import LOGGER
from config import UPLOAD_PATH

app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],

)

MODEL = CustomOperator()
MILVUS_CLI = MilvusHelper()
MYSQL_CLI = MySQLHelper()


class Item(BaseModel):
    Table: Optional[str] = None
    File: str

# Define the interface to obtain raw pictures
@app.get('/data')
def image_path(image_path):
    # Get the image file
    try:
        LOGGER.info(f"Successfully load image: {image_path}")
        return FileResponse(image_path)
    except Exception as e:
        LOGGER.error(f"Upload image error: {e}")
        return {'status': False, 'msg': e}, 400


@app.get('/progress')
def get_progress():
    # Get the progress of dealing with images
    try:
        cache = Cache('./tmp')
        return f"current: {cache['current']}, total: {cache['total']}"
    except Exception as e:
        LOGGER.error(f"Error to get the progress: {e}")
        return {'status': False, 'msg': e}, 400


@app.post('/img/load')
async def load_images(item: Item):
    # Insert all the image under the file path to Milvus/MySQL
    try:
        total_num = do_load(item.Table, item.File, MODEL, MILVUS_CLI, MYSQL_CLI)
        LOGGER.info(f"Successfully loaded data, total objects: {total_num}")
        return "Successfully loaded data!"
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


@app.post('/img/search')
async def search_images(image: UploadFile = File(...), table_name: str = None):
    # Search the upload image in Milvus/MySQL
    try:
        # Save the upload image to server.
        content = await image.read()
        print('read pic succ')
        tmp_dir = os.path.join(UPLOAD_PATH, image.filename.split('.')[0])
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
            LOGGER.info(f"Mkdir the path: {tmp_dir}")
        img_path = os.path.join(tmp_dir, image.filename.lower())
        with open(img_path, "wb+") as f:
            f.write(content)
        img_name = os.listdir(tmp_dir)[0]
        search_img = os.path.join(tmp_dir, img_name)
        paths, distances = do_search(table_name, search_img, MODEL, MILVUS_CLI, MYSQL_CLI)
        #print(paths, distances)
        res = dict(zip(paths, distances))
        res = sorted(res.items(), key=lambda item: item[1])
        LOGGER.info("Successfully searched similar images!")
        return res
    except Exception as e:
        LOGGER.error(e)
        return {'status': False, 'msg': e}, 400


if __name__ == '__main__':
    uvicorn.run(app=app, host='0.0.0.0', port=5000)
