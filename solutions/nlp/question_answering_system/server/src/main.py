import os
import uvicorn
from fastapi import FastAPI, File, UploadFile
from starlette.middleware.cors import CORSMiddleware

from config import UPLOAD_PATH
from logs import LOGGER
from milvus_helpers import MilvusHelper
from mysql_helpers import MySQLHelper
from encode import SentenceModel
from operations.load import do_load
from operations.search import do_search, do_get_answer
from operations.count import do_count
from operations.drop import do_drop

app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,

    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],

)

MODEL = SentenceModel()
MILVUS_CLI = MilvusHelper()
MYSQL_CLI = MySQLHelper()

# Mkdir '/tmp/qa-data'
if not os.path.exists(UPLOAD_PATH):
    os.makedirs(UPLOAD_PATH)


@app.post('/qa/load_data')
async def do_load_api(file: UploadFile = File(...), table_name: str = None):
    try:
        text = await file.read()
        fname_path = os.path.join(UPLOAD_PATH, file.filename)
        with open(fname_path, 'wb') as f:
            f.write(text)
        total_num = do_load(table_name, fname_path, MODEL, MILVUS_CLI, MYSQL_CLI)
        LOGGER.info(f"Successfully loaded data, total count: {total_num}")
        return {'status': True, 'msg': f"Successfully loaded data: {total_num}"}, 200
    except Exception as e:
        LOGGER.error(e)
        return {'status': False, 'msg': e}, 400


@app.get('/qa/search')
async def do_get_question_api(question: str, table_name: str = None):
    try:
        questions, _= do_search(table_name, question, MODEL, MILVUS_CLI, MYSQL_CLI)
        LOGGER.info("Successfully searched similar images!")
        return {'status': True, 'msg': questions}, 200
    except Exception as e:
        LOGGER.error(e)
        return {'status': False, 'msg': e}, 400


@app.get('/qa/answer')
async def do_get_answer_api(question: str, table_name: str = None):
    try:
        results = do_get_answer(table_name, question, MYSQL_CLI)
        return {'status': True, 'msg': results}
    except Exception as e:
        LOGGER.error(e)
        return {'status': False, 'msg': e}


@app.post('/qa/count')
async def count_images(table_name: str = None):
    try:
        num = do_count(table_name, MILVUS_CLI)
        LOGGER.info("Successfully count the number of questions!")
        return num
    except Exception as e:
        LOGGER.error(e)
        return {'status': False, 'msg': e}, 400


@app.post('/qa/drop')
async def drop_tables(table_name: str = None):
    try:
        status = do_drop(table_name, MILVUS_CLI, MYSQL_CLI)
        LOGGER.info("Successfully drop tables in Milvus and MySQL!")
        return {'status': True, 'msg': status}
    except Exception as e:
        LOGGER.error(e)
        return {'status': False, 'msg': e}, 400

if __name__ == '__main__':
    uvicorn.run(app=app, host='0.0.0.0', port=8000)
