import os
import os.path as path
from common.config import DATA_PATH, DEFAULT_TABLE
from common.config import UPLOAD_PATH
from common.config import DEFAULT_CACHE_DIR
from service.train import do_train
from service.search import do_search
from service.count import do_count
from service.delete import do_delete
from service.theardpool import thread_runner
from flask_cors import CORS
from flask import Flask, request, send_file, jsonify
from flask_restful import reqparse
from werkzeug.utils import secure_filename
import tensorflow.compat.v1 as tf
from tensorflow.python.keras.backend import set_session
from diskcache import Cache
import shutil
from preprocessor.resnet50 import Resnet50

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.5
global sess
sess = tf.Session(config=config)
set_session(sess)

app = Flask(__name__)
# ALLOWED_EXTENSIONS = set(['jpg', 'png'])
# app.config['UPLOAD_FOLDER'] = UPLOAD_PATH
# app.config['JSON_SORT_KEYS'] = False
CORS(app)

model = None


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
    cache = Cache(DEFAULT_CACHE_DIR)
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
        if not os.path.exists(UPLOAD_PATH):
            os.mkdir(UPLOAD_PATH)
        file_path = os.path.join(UPLOAD_PATH, filename)
        file.save(file_path)
        res_id, res_distance = do_search(table_name, file_path, top_k, Resnet50())
        if isinstance(res_id, str):
            return res_id
        res_img = [request.url_root + "data/" + x for x in res_id]
        res = dict(zip(res_img, res_distance))
        res = sorted(res.items(), key=lambda item: item[1])
        return jsonify(res), 200
    return "not found", 400


if __name__ == "__main__":
    app.run(host="0.0.0.0")
