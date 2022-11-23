import os
from typing import Optional

import gdown
import torch
import torch.nn as nn
import uvicorn
from diskcache import Cache
from fastapi import FastAPI
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

from milvus3d.MeshNet import MeshNet
from config import UPLOAD_PATH
from config import WEIGHTS, CUDA_DEVICE
from logs import LOGGER
from milvus_helpers import MilvusHelper
from mysql_helpers import MySQLHelper
from operations.count import do_count
from operations.drop import do_drop
from operations.load import do_load
from operations.search import do_search
from milvus3d.transform import Transformer

app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],

)
weights_dir = "../data/models/"
weights = "MeshNet_best_9192.pkl"
if not os.path.exists(weights_dir):
    os.mkdir(weights_dir)
    if not os.path.exists(weights_dir + weights):
        gdown.download("https://drive.google.com/uc?id=1t5jyJ4Ktmlck6GYhNTPVTFZuRP7wPUYq", weights_dir + weights)


os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_DEVICE

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = MeshNet(require_fea=False)
model = nn.DataParallel(model)

model.load_state_dict(torch.load(WEIGHTS, map_location=device))
model.to(device)

model.eval()
transformer = Transformer(model)

MILVUS_CLI = MilvusHelper()
MYSQL_CLI = MySQLHelper()

# Mkdir '/tmp/search-models'
if not os.path.exists(UPLOAD_PATH):
    os.makedirs(UPLOAD_PATH)
    LOGGER.info("mkdir the path:{} ".format(UPLOAD_PATH))


LOGGER.info(
    """
    
    $$\      $$\ $$\ $$\                                       $$$$$$\  $$$$$$$\  
    $$$\    $$$ |\__|$$ |                                     $$ ___$$\ $$  __$$\ 
    $$$$\  $$$$ |$$\ $$ |$$\    $$\ $$\   $$\  $$$$$$$\       \_/   $$ |$$ |  $$ |
    $$\$$\$$ $$ |$$ |$$ |\$$\  $$  |$$ |  $$ |$$  _____|        $$$$$ / $$ |  $$ |
    $$ \$$$  $$ |$$ |$$ | \$$\$$  / $$ |  $$ |\$$$$$$\          \___$$\ $$ |  $$ |
    $$ |\$  /$$ |$$ |$$ |  \$$$  /  $$ |  $$ | \____$$\       $$\   $$ |$$ |  $$ |
    $$ | \_/ $$ |$$ |$$ |   \$  /   \$$$$$$  |$$$$$$$  |      \$$$$$$  |$$$$$$$  |
    \__|     \__|\__|\__|    \_/     \______/ \_______/        \______/ \_______/ 
    
    Welcome to Milvus 3D! :)
    
    Version 0.1

    Author: Sida Shen                                                                            
    """
)


# # Define the interface to obtain raw pictures
# @app.get('/data')
# def get_model(model_path):
#     # Get the image file
#     try:
#         LOGGER.info(("Successfully load model: {}".format(model_path)))
#         return FileResponse(model_path)
#     except Exception as e:
#         LOGGER.error("upload model error: {}".format(e))
#         return {'status': False, 'msg': e}, 400


@app.get('/progress')
def get_progress():
    # Get the progress of 3d model
    try:
        cache = Cache('./tmp')
        return "current: {}, total: {}".format(cache['current'], cache['total'])
    except Exception as e:
        LOGGER.error("upload image error: {}".format(e))
        return {'status': False, 'msg': e}, 400


class Item(BaseModel):
    Table: Optional[str] = None
    File: str


@app.post('/img/load')
async def load_models(item: Item):
    # Insert all the image under the file path to Milvus/MySQL
    try:
        total_num = do_load(item.Table, item.File, transformer, MILVUS_CLI, MYSQL_CLI)
        LOGGER.info("Successfully loaded data, total count: {}".format(total_num))
        return "Successfully loaded data, total count: {}".format(total_num)
    except Exception as e:
        LOGGER.error(e)
        return {'status': False, 'msg': e}, 400


@app.post('/img/search')
async def search_images(model_path: str, table_name: str = None):
    # Search the upload image in Milvus/MySQL
    # Search the upload image in Milvus/MySQL
    try:
        paths, distances = do_search(table_name, model_path, transformer, MILVUS_CLI, MYSQL_CLI)
        res = dict(zip(paths, distances))
        res = sorted(res.items(), key=lambda item: item[1])
        LOGGER.info("Successfully searched similar models!")
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
