import uvicorn
import os
from fastapi import FastAPI, File, UploadFile
from starlette.middleware.cors import CORSMiddleware
from milvus_helpers import MilvusHelper
from mysql_helpers import MySQLHelper
from operations.load import do_load
from operations.search import do_search, do_get_answer
from operations.count import do_count
from operations.drop import do_drop
from logs import LOGGER
from encode import Sentence_model


app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,

    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],

)

MODEL = Sentence_model()
MILVUS_CLI = MilvusHelper()
MYSQL_CLI = MySQLHelper()


@app.post('/qa/load_data')
async def do_load_api(file: UploadFile = File(...), table_name: str = None):
    try:
        text = await file.read()
        fname = file.filename
        dirs = "QA_data"
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        fname_path = os.path.join(os.getcwd(), os.path.join(dirs, fname))
        with open(fname_path, 'wb') as f:
            f.write(text)
    except Exception as e:
        return {'status': False, 'msg': 'Failed to load data.'}
    try:
        total_num = do_load(table_name, fname_path, MODEL, MILVUS_CLI, MYSQL_CLI)
        LOGGER.info("Successfully loaded data, total count: {}".format(total_num))
        return {'status': True, 'msg': "Successfully loaded data: {}".format(total_num)}, 200
    except Exception as e:
        LOGGER.error(e)
        return {'status': False, 'msg': e}, 400


@app.get('/qa/search')
async def do_get_question_api(question: str, table_name: str = None):
    try:
        questions, distances = do_search(table_name, question, MODEL, MILVUS_CLI, MYSQL_CLI)
        res = dict(zip(questions, distances))
        # res = sorted(res.items(), key=lambda item: item[1])
        LOGGER.info("Successfully searched similar images!")
        return {'status': True, 'msg': res}, 200
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
    # Returns the total number of questions in the system
    try:
        num = do_count(table_name, MILVUS_CLI)
        LOGGER.info("Successfully count the number of questions!")
        return num
    except Exception as e:
        LOGGER.error(e)
        return {'status': False, 'msg': e}, 400


@app.post('/qa/drop')
async def drop_tables(table_name: str = None):
    # Delete the collection of Milvus and MySQL
    try:
        msg = do_drop(table_name, MILVUS_CLI, MYSQL_CLI)
        LOGGER.info("Successfully drop tables in Milvus and MySQL!")
        return {'status': True, 'msg': msg}, 200
    except Exception as e:
        LOGGER.error(e)
        return {'status': False, 'msg': e}, 400

if __name__ == '__main__':
    uvicorn.run(app=app, host='0.0.0.0', port=8000)
