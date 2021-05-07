import os
import sys
import torch
from test_config import config
from config import data_config, network_config
from flask_restful import reqparse
from flask import Flask, request, send_file, jsonify
from service.theardpool import thread_runner
from service.train import do_train
from service.search import do_search, query_name_from_ids
from common.config import DATA_PATH, DEFAULT_TABLE
import shutil
import torchvision.transforms as transforms
from werkzeug.utils import secure_filename
import pickle
from flask_cors import CORS
from common.const import UPLOAD_PATH

app = Flask(__name__)
ALLOWED_EXTENSIONS = set(['jpg', 'png'])
app.config['UPLOAD_FOLDER'] = UPLOAD_PATH
app.config['JSON_SORT_KEYS'] = False
CORS(app)


def load_model():
    # global graph
    # graph = tf.get_default_graph()
    global model
    model_file = args.model_path
    model, _ = network_config(args, 'test', None, True, model_file, True)


@app.route('/api/v1/train', methods=['POST'])
def do_train_api():
    input_args = reqparse.RequestParser(). \
        add_argument('Table', type=str). \
        parse_args()
    table_name = input_args['Table']
    print(table_name)
    # file_path = input_args['File']
    try:
        thread_runner(1, do_train, table_name, test_loader, model, args)
        # filenames = os.listdir(file_path)
        # if not os.path.exists(DATA_PATH):
        #     os.mkdir(DATA_PATH)
        # for filename in filenames:
        #     shutil.copy(file_path + '/' + filename, DATA_PATH)
        return "Start"
    except Exception as e:
        return "Error with {}".format(e)


@app.route('/api/v1/search', methods=['POST'])
def do_search_api():
    input_args = reqparse.RequestParser(). \
        add_argument("Table", type=str). \
        add_argument("Num", type=int, default=1). \
        add_argument("Caption", type=str). \
        parse_args()

    captions = input_args['Caption']
    print(captions)
    table_name = input_args['Table']
    top_k = input_args['Num']
    if not table_name:
        table_name = DEFAULT_TABLE
        print(table_name)
    if not captions:
        return "no captions", 400
    if captions:
        # filename = secure_filename(file.filename)
        # file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        # file.save(file_path)
        res_id, res_distance = do_search(table_name, captions, top_k, model, args)
        if isinstance(res_id, str):
            return res_id
        # res_img = [request.url_root + "data/" + x for x in res_id]
        # res = dict(zip(res_img, res_distance))
        # res = sorted(res.items(), key=lambda item: item[1])
        # return jsonify(res), 200
    return "not found", 400

if __name__ == "__main__":
    global args
    global test_loader
    global test_transform
    args = config()
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    test_loader = data_config(args.image_dir, args.anno_dir, 64, 'test', args.max_length, test_transform)
    load_model()
    # do_train('test02', test_loader, model, args)
    captions = 'The man is wearing a blue and white striped tank top and green pants. He has pink headphones around his neck.'
    do_search('test02', captions, 10, model, args)
    app.run(host="192.168.1.85", port='5001')
