from fastapi import Depends, FastAPI, File, UploadFile

import os
import pandas as pd
import os.path
import logging
from fastapi.middleware.cors import CORSMiddleware
from bert_serving.client import BertClient
from pydantic import BaseModel
from starlette.responses import FileResponse
import numpy as np
import random

from src.get_answer import load_data, get_similar_question, get_result
from src.config import BERT_HOST, BERT_PORT
from src.milvus_operating import milvus_client
from src.pg_operating import connect_postgres_server

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post('/qa/load')
async def do_load_api(file: UploadFile = File(...)):
    try:
        text = await file.read()
        fname = file.filename
        dirs = "QA_data/"
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        fname_path = dirs + "/" + fname
        with open(fname_path, 'wb') as f:
            f.write(text)
    except Exception as e:
        return {'status': False, 'msg': 'Failed to load data.'}
    try:
        conn = connect_postgres_server()
        cursor = conn.cursor()
        client = milvus_client()
        bc = BertClient(ip=BERT_HOST, port=BERT_PORT, check_length=False)
        status, message = load_data(fname_path, client, conn, cursor, bc)
        return {'status': status, 'msg': message}
    except Exception as e:
        print("load data faild: ", e)
        return {'status': False, 'msg': 'Failed to load data.'}
    finally:
        cursor.close()
        conn.close()
        bc.close()


@app.get('/qa/search')
async def do_get_question_api(question: str):
    if not question:
        return {'status': False, 'msg': 'Please enter the query.'}
    if question:
        try:
            # user_id = 'qa_' + user_id
            conn = connect_postgres_server()
            cursor = conn.cursor()
            client = milvus_client()

            bc = BertClient(ip=BERT_HOST, port=BERT_PORT, check_length=False)

            output = get_similar_question(question, client, conn, cursor, bc)
            if output:
                return {'status': True, 'msg': output}
            else:
                return {'status': False, 'msg': 'No similar questions in the database'}
        except Exception as e:
            print('search faild: ', e)
            return {'status': False, 'msg': 'Failed to search, please try again.'}
        finally:
            cursor.close()
            conn.close()
            bc.close()
    return {'status': False, 'msg': 'Failed to search, please try again.'}


@app.get('/qa/answer')
async def do_get_answer_api(question: str):
    if not question:
        return {'status': False, 'msg': 'Please enter the query.'}
    if question:
        try:
            # user_id = 'qa_' + user_id
            conn = connect_postgres_server()
            cursor = conn.cursor()
            results = get_result(question, conn, cursor)
            if results:
                return {'status': True, 'msg': results[0][0]}
            else:
                return {'status': False, 'msg': 'There is no answer to this question in the database'}
        except Exception as e:
            print('get answer faild: ', e)
            return {'status': False, 'msg': 'Failed to search, please try again.'}
        finally:
            cursor.close()
            conn.close()
    return {'status': False, 'msg': 'Failed to search, please try again.'}
