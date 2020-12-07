import os
import logging
from service.insert import do_insert_logo, do_insert_face
from service.search import do_search_logo, do_search_face, do_only_her
from service.count import do_count_table
from service.delete import do_delete_table
from service.format import format_info
from indexer.index import milvus_client
from indexer.tools import connect_mysql
from common.config import UPLOAD_PATH
import time
from fastapi import FastAPI
from fastapi import File
from fastapi import UploadFile
import uvicorn
from starlette.responses import FileResponse
from starlette.requests import Request
import uuid
from starlette.middleware.cors import CORSMiddleware
from resnet50_encoder.encode import CustomOperator
from face_encoder.deploy.encode import Encode

app = FastAPI()
image_encoder = CustomOperator()
face_encoder = Encode()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])


def init_conn():
    conn = connect_mysql()
    cursor = conn.cursor()
    index_client = milvus_client()
    return index_client, conn, cursor


@app.get('/countTable')
async def do_count_images_api(table_name: str = None):
    try:
        index_client, conn, cursor = init_conn()
        rows_milvus, rows_mysql = do_count_table(index_client, conn, cursor, table_name)
        return "{0},{1}".format(rows_milvus, rows_mysql), 200
    except Exception as e:
        logging.error(e)
        return "Error with {}".format(e), 400


@app.delete('/deleteTable')
async def do_delete_table_api(table_name: str = None):
    try:
        index_client, conn, cursor = init_conn()
        status = do_delete_table(index_client, conn, cursor, table_name)
        return "{}".format(status)
    except Exception as e:
        logging.error(e)
        return "Error with {}".format(e), 400


@app.get('/getImage')
async def image_endpoint(img: str):
    try:
        print("load img:", img)
        return FileResponse(img, media_type="image/jpg")
    except Exception as e:
        logging.error(e)
        return None, 200


@app.post('/insertFace')
async def do_insert_face_api(name: str, image: UploadFile = File(...), info: str = None, table_name: str = None):
    try:
        content = await image.read()
        filename = UPLOAD_PATH + "/" + uuid.uuid4().hex + ".jpg"
        with open(filename, 'wb') as f:
            f.write(content)
        index_client, conn, cursor = init_conn()
        info = do_insert_face(image_encoder, index_client, conn, cursor, table_name, filename, name, info)
        return info, 200
    except Exception as e:
        logging.error(e)
        return "Error with {}".format(e), 400


@app.post('/insertLogo')
async def do_insert_logo_api(name: str, image: UploadFile = File(...), info: str = None, table_name: str = None):
    try:
        content = await image.read()
        filename = UPLOAD_PATH + "/" + uuid.uuid4().hex + ".jpg"
        with open(filename, 'wb') as f:
            f.write(content)
        index_client, conn, cursor = init_conn()
        info = do_insert_logo(image_encoder, index_client, conn, cursor, table_name, filename, name, info)
        return info, 200
    except Exception as e:
        logging.error(e)
        return "Error with {}".format(e), 400


@app.post('/getLogoInfo')
async def get_item_info(request: Request, video: UploadFile = File(...), table_name: str = None):
    try:
        content = await video.read()
        filename = UPLOAD_PATH + "/" + video.filename
        with open(filename, "wb") as f:
            f.write(content)

        index_client, conn, cursor = init_conn()
        host = request.headers['host']
        info, times = do_search_logo(image_encoder, index_client, conn, cursor, table_name, filename)
        result_dic = {"code": 0, "msg": "success"}

        results = []
        for i in range(len(info)):
            re = {
                "milvus_id": info[i][0],
                "obj_name": info[i][1],
                "obj_info": info[i][2],
                "obj_image": "http://" + str(host) + "/getImage?img=" + info[i][3],
                "time": times[i].split("-")[0]
            }
            results.append(re)
        result_dic["data"] = results
        return result_dic, 200
    except Exception as e:
        logging.error(e)
        return "Error with {}".format(e), 400


@app.post('/WhoInVideo')
async def who_in_video(request: Request, video: UploadFile = File(...), table_name: str = None):
    try:
        content = await video.read()
        filename = UPLOAD_PATH + "/" + video.filename
        with open(filename, "wb") as f:
            f.write(content)

        index_client, conn, cursor = init_conn()
        host = request.headers['host']
        info = do_only_her(face_encoder, index_client, conn, cursor, table_name, filename)
        result_dic = {"code": 0, "msg": "success"}
        time_of_occur = {}
        time = 1
        for frame in info:
            for i in range(len(frame)):
                if not frame[i][1] in time_of_occur:
                    time_of_occur[frame[i][1]] = ["http://" + str(host) + "/getImage?img=" + frame[i][3], frame[i][2],
                                                  [time]]
                else:
                    time_of_occur[frame[i][1]][2].append(time)
            time += 1
        results = format_info(time_of_occur)
        result_dic["data"] = results
        return result_dic, 200
    except Exception as e:
        logging.error(e)
        return "Error with {}".format(e), 400


@app.post('/getFaceInfo')
async def get_face_info(request: Request, image: UploadFile = File(...), table_name: str = None):
    try:
        content = await image.read()
        filename = UPLOAD_PATH + "/" + image.filename
        with open(filename, "wb") as f:
            f.write(content)
        index_client, conn, cursor = init_conn()
        host = request.headers['host']
        info = do_search_face(face_encoder, index_client, conn, cursor, table_name, filename)
        result_dic = {"code": 0, "msg": "success"}

        results = []
        for i in range(len(info)):
            re = {
                "milvus_id": info[i][0],
                "face_name": info[i][1],
                "face_info": info[i][2],
                "face_image": "http://" + str(host) + "/getImage?img=" + info[i][3]
            }
            results.append(re)
        result_dic["data"] = results
        return result_dic, 200
    except Exception as e:
        logging.error(e)
        return "Error with {}".format(e), 400


if __name__ == '__main__':
    uvicorn.run(app=app, host='192.168.1.58', port=8000)
