import uvicorn
import os
from fastapi import FastAPI, File, UploadFile, Response
from fastapi.responses import StreamingResponse
from diskcache import Cache
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse
from starlette.requests import Request
from subprocess import call
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
from config import UPLOAD_PATH, DISTANCE_LIMIT, DEFAULT_TABLE

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

def convert_avi_to_mp4(avi_file_path):
    try:
        new_path = avi_file_path[:-4] + '.mp4'
        os.popen(
            "ffmpeg -i {input} -ac 2 -b:v 2000k -c:a aac -c:v libx264 -b:a 160k -vprofile high -bf 0 -strict experimental -f mp4 {output}".format(
                input=avi_file_path, output=new_path))
        return {'status': True, 'msg': 'Successfully converted avi to mp4!'}
    except Exception as e:
        logging.error(e)
        return None, 200

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

@app.get('/video/getVideo')
async def video_endpoint(video: str, response: Response):
    try:
        filelike = open(video, mode = "rb")
        return StreamingResponse(filelike, media_type="video/mp4", headers={"Accept-Ranges": "bytes"})
    except Exception as e:
        LOGGER.error(e)
        return {'status': False, 'msg': e}

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
async def load_image(item: Item):
    # Insert all the image under the file path to Milvus/MySQL
    try:
        TABLE_NAME = DEFAULT_TABLE
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
        if table_name == None:
            table_name = DEFAULT_TABLE
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
        if table_name == None:
            table_name = DEFAULT_TABLE
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
        if table_name == None:
            table_name = DEFAULT_TABLE
        # Save the upload image to server.
        content = await video.read()
        video_path = os.path.join(UPLOAD_PATH, video.filename)
        convert_avi_to_mp4(video_path)
        with open(video_path, "wb+") as f:
            f.write(content)
        host = request.headers['host']
        paths, objects, distances, times = do_search(table_name, video_path, MODEL, MILVUS_CLI, MYSQL_CLI)
        res = ["http://" + str(host) + "/video/getVideo?video=" + video_path.replace(".avi", ".mp4")]
        #res = []
        for i in range(len(paths)):
            if DISTANCE_LIMIT != None:
                if float(distances[i]) < DISTANCE_LIMIT:
                    re = {
                        "object": objects[i],
                        "image": "http://" + str(host) + "/data?image_path=" + paths[i],
                        "distance": distances[i],
                        "time": times[i]
                        }
                else:
                    re = {
                        "object": None,
                        "image": None,
                        "distance": None,
                        "time": None
                        }
            else:
                re = {
                    "object": objects[i],
                    "image": "http://" + str(host) + "/data?image_path=" + paths[i],
                    "distance": distances[i],
                    "time": times[i]
                    }
            if re["object"] != None:
                res.append(re)
        LOGGER.info("Successfully searched similar images!")
        #print(len(res))
        return res
    except Exception as e:
        LOGGER.error(e)
        return {'status': False, 'msg': e}, 400

if __name__ == '__main__':
    uvicorn.run(app=app, host='0.0.0.0', port=5000)
