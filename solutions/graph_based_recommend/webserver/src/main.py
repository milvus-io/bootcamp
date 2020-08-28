import os
import logging
from service.search import do_search, get_list_info, get_ids_info
from service.count import do_count
from service.delete import do_delete_table
from indexer.index import milvus_client
from indexer.tools import connect_mysql
from common.config import OUT_PATH
import time
from fastapi import FastAPI
import uvicorn
from starlette.responses import FileResponse
from starlette.requests import Request
import random
from starlette.middleware.cors import CORSMiddleware

app = FastAPI()

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


def get_img_list():
    list_id = []
    list_ids = os.listdir(OUT_PATH)
    list_ids.sort()
    return list_ids


@app.get('/countTable')
async def do_count_images_api(table_name: str=None):
    try:
        index_client, conn, cursor = init_conn()
        rows_milvus, rows_mysql = do_count(index_client, conn, cursor, table_name)
        return "{0},{1}".format(rows_milvus, rows_mysql), 200
    except Exception as e:
        logging.error(e)
        return "Error with {}".format(e), 400


@app.delete('/deleteTable')
async def do_delete_table_api(table_name: str=None):
    try:
        index_client, conn, cursor = init_conn()
        status = do_delete_table(index_client, conn, cursor, table_name)
        return "{}".format(status)
    except Exception as e:
        logging.error(e)
        return "Error with {}".format(e), 400


@app.get('/getImage')
def image_endpoint(img: int):
    try:
        img_path = OUT_PATH + '/' + str(img) + '.jpg'
        print(img_path)
        return FileResponse(img_path, media_type="image/jpg")
    except Exception as e:
        logging.error(e)
        return None, 200


@app.post('/getRandom')
def get_random_item(request: Request, num: int=None, table_name: str=None):
    try:
        if not num:
            num = 16
        index_client, conn, cursor = init_conn()
        img_list = get_img_list()
        list_id = random.sample(img_list, num)
        host = request.headers['host']
        print(list_id)
        info = get_list_info(conn, cursor, table_name, host, list_id)
        return info, 200
    except Exception as e:
        logging.error(e)
        return "Error with {}".format(e), 400


@app.get('/getInfo')
def get_item_info(request: Request, ids: int, table_name: str=None):
    try:
        index_client, conn, cursor = init_conn()
        host = request.headers['host']
        info = get_ids_info(conn, cursor, table_name, host, ids)
        return info, 200
    except Exception as e:
        logging.error(e)
        return "Error with {}".format(e), 400


@app.post('/getSimilarUser')
def do_search_images_api(request: Request, search_id: list, table_name: str=None):
    try:
        index_client, conn, cursor = init_conn()
        host = request.headers['host']
        img_list = get_img_list()
        list_id = do_search(index_client, conn, cursor, img_list, search_id, table_name)
        # print("--------", list_id)
        info = get_list_info(conn, cursor, table_name, host, list_id)
        return info, 200

    except Exception as e:
        logging.error(e)
        return "Error with {}".format(e), 400


if __name__ == '__main__':
    uvicorn.run(app=app, host='127.0.0.1', port=8000)