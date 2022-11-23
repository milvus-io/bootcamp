import uvicorn
import os
import os.path
from logs import LOGGER
from fastapi import FastAPI, File, UploadFile
from starlette.middleware.cors import CORSMiddleware
from milvus_helpers import MilvusHelper
from mysql_helpers import MySQLHelper
from operations.load import import_data
from operations.search import search_in_milvus
from operations.count import do_count
from operations.drop import do_drop
from pydantic import BaseModel
from typing import Optional



app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])
MILVUS_CLI = MilvusHelper()
MYSQL_CLI = MySQLHelper()

class Item(BaseModel):
    Table: Optional[str] = None
    File:str

@app.post('/text/count')
async def count_text(table_name: str = None):
    # Return the total number of data in the system
    try:
        num = do_count(table_name, MILVUS_CLI)
        LOGGER.info("Successfully count the number of sequences!")
        return num
    except Exception as e:
        LOGGER.error(e)
        return {'status': False, 'msg': e}, 400


@app.post('/text/drop')
async def drop_tables(table_name: str = None):
    # Delete the collection in Milvus and table in MySQL
    try:
        status = do_drop(table_name, MILVUS_CLI, MYSQL_CLI)
        LOGGER.info("Successfully drop tables in Milvus and MySQL!")
        return status
    except Exception as e:
        LOGGER.error(e)
        return {'status': False, 'msg': e}, 400


@app.post('/text/load')
async def load_text(file: UploadFile = File(...), table_name: str = None):
    # Create collection in Milvus & table in Mysql, and insert data
    try:
        text = await file.read()
        fname = file.filename
        dirs = "data"
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        fname_path = os.path.join(os.getcwd(), os.path.join(dirs, fname))
        with open(fname_path, 'wb') as f:
            f.write(text)
    except Exception :
        return {'status': False, 'msg': 'Failed to load data.'}
    # Insert data under the file path to Milvus & MySQL
    try:
        total_num = import_data(table_name, fname_path ,MILVUS_CLI, MYSQL_CLI)
        LOGGER.info("Vectorizer is saved!")
        LOGGER.info(f"Successfully loaded data, total count: {total_num}")
        return "Successfully loaded data!"
    except Exception as e:
        LOGGER.error(e)
        return {'status': False, 'msg': e}, 400


@app.get('/text/search')
async def do_search_api(table_name: str = None, query_sentence: str = None):
    # Search for target DNA sequence and return class & distance
    try:

        ids, results_classes, seq_genes, distances = search_in_milvus(table_name,query_sentence, MILVUS_CLI, MYSQL_CLI)
        res = []
        for i, c, s, d in zip(ids, results_classes, seq_genes, distances):
            dicts = {'milvus_id': i,'seq_class': c, 'seq_gene': s, 'IP distance':d}
            res+=[dicts]
        LOGGER.info("Successfully searched similar sequence!")
        return res
    except Exception as e:
        LOGGER.error(e)
        return {'status': False, 'msg': e}, 400


if __name__ == "__main__":
    uvicorn.run(app=app, host='0.0.0.0', port=5001)
