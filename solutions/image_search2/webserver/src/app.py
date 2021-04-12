import os
import os.path as path
import logging
from common.config import DATA_PATH, DEFAULT_TABLE
from common.const import UPLOAD_PATH
from common.const import input_shape
from common.const import default_cache_dir
from service.train import do_train
from service.search import do_search
from service.count import do_count
from service.delete import do_delete
from service.theardpool import thread_runner
# from preprocessor.vggnet import vgg_extract_feat
from indexer.index import milvus_client, create_table, insert_vectors, delete_table, search_vectors, create_index
from service.search import query_name_from_ids
from flask_cors import CORS
from flask import Flask, request, send_file, jsonify
from flask_restful import reqparse
from werkzeug.utils import secure_filename
from keras.preprocessing import image
import numpy as np
import tensorflow as tf
from numpy import linalg as LA
from resnet50_encoder.encode import CustomOperator
from diskcache import Cache
import shutil
from itertools import groupby
from operator import itemgetter


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.5
image_encoder = CustomOperator()
# global sess
# sess = tf.Session(config=config)
# set_session(sess)

app = Flask(__name__)
ALLOWED_EXTENSIONS = set(['jpg', 'png'])
app.config['UPLOAD_FOLDER'] = UPLOAD_PATH
app.config['JSON_SORT_KEYS'] = False
CORS(app)



@app.route('/api/v1/train', methods=['POST'])
def do_train_api():
    args = reqparse.RequestParser(). \
        add_argument('Table', type=str). \
        add_argument('File', type=str). \
        parse_args()
    table_name = args['Table']
    file_path = args['File']
    try:
        thread_runner(1, do_train, table_name, file_path)
        #print(file_path)
        filenames = os.listdir(file_path)
        if not os.path.exists(DATA_PATH):
            os.mkdir(DATA_PATH)
        for filename in filenames:
            shutil.copy(file_path + '/' + filename, DATA_PATH)
        return "Start"
    except Exception as e:
        return "Error with {}".format(e)


@app.route('/api/v1/delete', methods=['POST'])
def do_delete_api():
    args = reqparse.RequestParser(). \
        add_argument('Table', type=str). \
        parse_args()
    table_name = args['Table']
    print("delete table.")
    status = do_delete(table_name)
    try:
        shutil.rmtree(DATA_PATH)
    except:
        print("cannot remove", DATA_PATH)
    return "{}".format(status)


@app.route('/api/v1/count', methods=['POST'])
def do_count_api():
    args = reqparse.RequestParser(). \
        add_argument('Table', type=str). \
        parse_args()
    table_name = args['Table']
    rows = do_count(table_name)
    return "{}".format(rows)


@app.route('/api/v1/process')
def thread_status_api():
    cache = Cache(default_cache_dir)
    return "current: {}, total: {}".format(cache['current'], cache['total'])


@app.route('/data/<image_name>')
def image_path(image_name):
    file_name = DATA_PATH + '/' + image_name
    if path.exists(file_name):
        return send_file(file_name)
    return "file not exist"


@app.route('/api/v1/search', methods=['POST'])
def do_search_api():
    args = reqparse.RequestParser(). \
        add_argument("Table", type=str). \
        add_argument("Num", type=int, default=1). \
        parse_args()

    table_name = args['Table']
    if not table_name:
        table_name = DEFAULT_TABLE
    top_k = args['Num']
    file = request.files.get('file', "")
    if not file:
        return "no file data", 400
    if not file.name:
        return "need file name", 400
    if file:
        filename = secure_filename(file.filename)
        img_path = app.config['UPLOAD_FOLDER']+'/'+filename.split('.')[0]
        if not os.path.exists(img_path):
            os.mkdir(img_path)
        file_path = os.path.join(img_path, filename)
        file.save(file_path)
        res_id,res_distance = do_search(image_encoder,table_name, img_path, top_k)
        # print(res_id,res_distance)
        # if isinstance(res_id, str):
        #     return res_id
        res ={}
        for img, dis in zip(res_id,res_distance):
           # print("[[[[[[[[[[[[[[", img, dis)
            key_img = request.url_root +"data/" + img
            if key_img not in res.keys() or res[key_img] > dis:
                res[key_img] = dis
        #print("dic res:", res)
        # res = dict(zip(res_img,res_distance))

        #res = [list(g)[0] for k, g in groupby(sorted(res), key=itemgetter(0))]
       # print(",,,,,,,",res)
        res = sorted(res.items(),key=lambda item:item[1])
        # myslice = slice(top_k)  
        # res=res[myslice] 
        return jsonify(res), 200
    return "not found", 400



if __name__ == "__main__":
    app.run(host="0.0.0.0")
