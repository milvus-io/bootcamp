import os
import uuid
import cv2
import uvicorn
from diskcache import Cache
from fastapi import FastAPI
from fastapi import File
from fastapi import UploadFile
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse
from operations.count import do_count
from operations.drop import do_drop
from operations.load import do_load
from operations.search import do_search
from milvus_helpers import MilvusHelper
from mysql_helpers import MySQLHelper
from frame_extract import FrameExtract
from logs import LOGGER
from config import UPLOAD_PATH
from encode import Resnet50

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
FRAME = FrameExtract()

# Mkdir '/tmp/search-images'
if not os.path.exists(UPLOAD_PATH):
    os.makedirs(UPLOAD_PATH)
    LOGGER.info("mkdir the path:{} ".format(UPLOAD_PATH))


@app.get('/data')
def video_path(gif_path):
    # Get the gif file
    try:
        LOGGER.info(("Successfully load gif: {}".format(gif_path)))
        return FileResponse(gif_path)
    except Exception as e:
        LOGGER.error("upload image error: {}".format(e))
        return {'status': False, 'msg': e}, 400


@app.get('/progress')
def get_progress():
    # Get the progress of dealing with images
    try:
        cache = Cache('./tmp')
        return "current: {}, total: {}".format(cache['current'], cache['total'])
    except Exception as e:
        LOGGER.error("upload image error: {}".format(e))
        return {'status': False, 'msg': e}, 400


@app.post('/video/count')
async def count_images(table_name: str = None):
    # Returns the total number of images in the system
    try:
        num = do_count(table_name, MILVUS_CLI, MYSQL_CLI)
        LOGGER.info("Successfully count the number of images!")
        return num
    except Exception as e:
        LOGGER.error(e)
        return {'status': False, 'msg': e}, 400


@app.post('/video/drop')
async def drop_tables(table_name: str = None):
    # Delete the collection of Milvus and MySQL
    try:
        status = do_drop(table_name, MILVUS_CLI, MYSQL_CLI)
        LOGGER.info("Successfully drop tables in Milvus and MySQL!")
        return status
    except Exception as e:
        LOGGER.error(e)
        return {'status': False, 'msg': e}, 400


@app.post('/video/load')
async def load_video(File: str, Table: str = None):
    # Insert all the video under the file path to Milvus/MySQL
    try:
        total_num = do_load(Table, File, MODEL, FRAME, MILVUS_CLI, MYSQL_CLI)
        LOGGER.info("Successfully loaded data, total count: {}".format(total_num))
        return {'status': True, 'msg': "Successfully loaded data!"}
    except Exception as e:
        LOGGER.error(e)
        return {'status': False, 'msg': e}, 400


@app.post('/video/search')
async def search_images(image: UploadFile = File(...), table_name: str = None):
    # Search the upload image in Milvus/MySQL
    try:
        # Save the upload image to server.
        content = await image.read()
        img_path = os.path.join(UPLOAD_PATH, image.filename)
        with open(img_path, "wb+") as f:
            f.write(content)
        paths, distances = do_search(table_name, img_path, MODEL, MILVUS_CLI, MYSQL_CLI)
        res = dict(zip(paths, distances))
        res = sorted(res.items(), key=lambda item: item[1])
        LOGGER.info("Successfully searched similar images!")
        return res
    except Exception as e:
        LOGGER.error(e)
        return {'status': False, 'msg': e}, 400


if __name__ == '__main__':
    uvicorn.run(app=app, host='0.0.0.0', port=5000)
