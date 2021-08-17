import uvicorn
import os
from fastapi import FastAPI, File, UploadFile
from diskcache import Cache
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse
from starlette.requests import Request
from typing import Optional
from milvus_helpers import MilvusHelper
from mysql_helpers import MySQLHelper
from operations.load import do_load
from operations.search import do_search
from operations.count import do_count
from operations.drop import do_drop
from encode import CustomOperator
from frame_extract import FrameExtract
from pydantic import BaseModel
from logs import LOGGER
from config import UPLOAD_PATH, DISTANCE_LIMIT

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


@app.get('/data')
def image_path(image_path):
    # Get the image file
    try:
        LOGGER.info(("Successfully load image: {}".format(image_path)))
        return FileResponse(image_path)
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


@app.post('/image/load')
async def load_video(item: Item):
    # Insert all the image under the file path to Milvus/MySQL
    try:
        TABLE_NAME = item.Table
        FILEPATH = item.File
        #print(TABLE_NAME, FILEPATH)
        total_num = do_load(TABLE_NAME, FILEPATH, MODEL, MILVUS_CLI, MYSQL_CLI)
        LOGGER.info("Successfully loaded data, total count: {}".format(total_num))
        return {'status': True, 'msg': "Successfully loaded data!"}
    except Exception as e:
        LOGGER.error(e)
        return {'status': False, 'msg': e}, 400


@app.post('/image/count')
async def count_videos(table_name: str = None):
    # Returns the total number of images in the system
    try:
        num = do_count(table_name, MILVUS_CLI)
        LOGGER.info("Successfully count the number of images!")
        return num
    except Exception as e:
        LOGGER.error(e)
        return {'status': False, 'msg': e}, 400


@app.post('/image/drop')
async def drop_tables(table_name: str = None):
    # Delete the collection of Milvus and MySQL
    try:
        status = do_drop(table_name, MILVUS_CLI, MYSQL_CLI)
        LOGGER.info("Successfully drop tables in Milvus and MySQL!")
        return status
    except Exception as e:
        LOGGER.error(e)
        return {'status': False, 'msg': e}, 400


@app.post('/video/search')
async def search_images(request: Request, video: UploadFile = File(...), table_name: str = None):
    # Search the upload image in Milvus/MySQL
    try:
        # Save the upload image to server.
        content = await video.read()
        video_path = os.path.join(UPLOAD_PATH, video.filename)
        with open(video_path, "wb+") as f:
            f.write(content)
        host = request.headers['host']
        paths, objects, distances = do_search(table_name, video_path, MODEL, MILVUS_CLI, MYSQL_CLI)
        res = []
        for i in range(len(paths)):
            if DISTANCE_LIMIT:
                if float(distances[i]) < DISTANCE_LIMIT:
                    re = {
                        "object": objects[i],
                        "image": "http://" + str(host) + "/getImage?img=" + paths[i],
                        "distance": distances[i]
                        }
                else:
                    re = {
                        "object": None,
                        "image": None,
                        "distance": None
                        }
            else:
                re = {
                    "object": objects[i],
                    "image": "http://" + str(host) + "/getImage?img=" + paths[i],
                    "distance": distances[i]
                    }
            res.append(re)
        LOGGER.info("Successfully searched similar images!")
        return res
    except Exception as e:
        LOGGER.error(e)
        return {'status': False, 'msg': e}, 400


if __name__ == '__main__':
    uvicorn.run(app=app, host='0.0.0.0', port=5000)
