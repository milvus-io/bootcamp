import os
import os.path as path
import logging

from flask_cors import CORS
from flask import Flask, request, send_file, jsonify
from flask_restful import reqparse
from werkzeug.utils import secure_filename

import numpy as np
from bert_serving.client import BertClient

import shutil
import urllib
import os
import time

# import simnet_milvus
from src.const import UPLOAD_PATH
import src.config as config

# import servive.milvus_toolkit as milvus_toolkit
from src.milvus_bert import search_in_milvus

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_PATH
app.config['JSON_SORT_KEYS'] = False
CORS(app)

bc = BertClient(timeout=10000)

@app.route('/api/v1/search', methods=['POST'])
def do_search_api():
    args = reqparse.RequestParser(). \
        add_argument("Table", type=str). \
        add_argument("query_text", type=str). \
        parse_args()

    table_name = args['Table']
    if not table_name:
        table_name = config.DEFAULT_TABLE
    query_sentence = args['query_text']
    if not query_sentence:
        return "no text"
    if query_sentence:
        try:
            output = search_in_milvus(table_name, query_sentence, bc)
            if output:
                return output
            else:
                return "未收录该问题"
        except Exception as e:
            return "Error with {}".format(e)
    return "not found", 400



if __name__ == "__main__":
    app.run(host="0.0.0.0")
