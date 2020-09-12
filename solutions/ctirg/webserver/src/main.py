import os
import traceback
import logging
from service.search import do_search
from service.count import do_count
from service.delete import do_delete_table
from indexer.index import milvus_client
from indexer.tools import connect_mysql
from common.config import OUT_PATH
import time
from fastapi import FastAPI
from fastapi import File
from fastapi import UploadFile
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
     host = "127.0.0.1"
     list_ids = os.listdir(OUT_PATH)
     list_ids.sort(key=lambda x: int(x[-10:-4]))
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



@app.get('/getImage')
def image_endpoint(img: str):
    try:
        img_path = OUT_PATH + '/' + img
        print(img_path)
        return FileResponse(img_path, media_type="image/png")
    except Exception as e:
        logging.error(e)
        return None, 200

@app.post('/getSimilarResult')
async def do_search_images_api(request: Request, Text: str, Image: UploadFile = File(...), table_name: str=None):
    try:
        index_client, conn, cursor = init_conn()
        host = request.headers['host']
        content = await Image.read()
        with open ("./test/test.png" ,'wb') as f :
            f.write(content)
        Image ="./test/test.png"
        img_list = get_img_list()
        list_id = do_search(index_client, conn, cursor ,Text, Image,table_name,img_list,host)
        print("--------", list_id)
        return list_id, 200
    except Exception as e:
        logging.error(e)
        print(traceback.format_exc())
       # return print(traceback.print_exc()),400
       # return " Error with {}".format(e), 400


if __name__ == '__main__':
    uvicorn.run(app=app, host='192.168.1.58', port=7000)
