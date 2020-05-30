import os
import os.path as path
import logging
from common.config import IMG_TABLE, VOC_TABLE
from common.const import UPLOAD_PATH, DATA_PATH
from service.search import do_search
from service.insert import do_insert
from service.count import do_count
from indexer.index import milvus_client, create_table, insert_vectors, delete_table, search_vectors, create_index
from flask_cors import CORS
from flask import Flask, request, send_file, jsonify, send_from_directory
from flask_restful import reqparse
from werkzeug.utils import secure_filename
import numpy as np
from numpy import linalg as LA
import datetime
from moviepy.editor import *


app = Flask(__name__, static_folder='static/build')
app.config['UPLOAD_FOLDER'] = UPLOAD_PATH
app.config['DATA_FOLDER'] = DATA_PATH
app.config['JSON_SORT_KEYS'] = False
CORS(app)

model = None

@app.route('/', defaults={'path': ''})

@app.route('/<path:path>')
def serve_static_index(path):
    print(app.static_folder + '/' + path)
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')


@app.route('/api/v1/count', methods=['POST'])
def do_count_api():
    args = reqparse.RequestParser(). \
        add_argument('Table', type=str). \
        parse_args()
    table_name = args['Table']
    rows = do_count(table_name)
    return "{}".format(rows)


@app.route('/api/v1/insert', methods=['POST'])
def do_insert_api():
    args = reqparse.RequestParser(). \
        add_argument("Name", type=str). \
        parse_args()

    name = args['Name']
    file_img = request.files.get('img', "")
    file_video = request.files.get('video', "")
    file_audio = request.files.get('audio', None)
    ids = '{0:%Y%m%d%H%M%S%f}'.format(datetime.datetime.now())
    status = {'status': 'faile', 'message':'there is no file data'}
    if file_video:
        filename = secure_filename(file_video.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file_video.save(file_path)
        video = VideoFileClip(file_path)
        audio = video.audio
        voc_path = os.path.join(app.config['UPLOAD_FOLDER'], ids[:-1] + '.wav')
        audio.write_audiofile(voc_path)
    elif file_audio:
        voc_path = os.path.join(app.config['UPLOAD_FOLDER'], ids[:-1] + '.wav')
        file_audio.save(voc_path)
    else:
        return jsonify(status), 200

    if file_img:
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], ids[:-1] + '.png')
        file_img.save(img_path)
        try:
            status = do_insert(name, ids[:-1], img_path, voc_path)
        except:
            status = {'status': 'faile', 'message':'please confirm only one face in camera'}
    else:
        return jsonify(status), 200
    return jsonify(status), 200


@app.route('/data/<image_name>')
def image_path(image_name):
    file_name = UPLOAD_PATH + '/' + image_name
    print(file_name)
    if path.exists(file_name):
        return send_file(file_name)
    return "file not exist"


@app.route('/api/v1/search', methods=['POST'])
def do_search_api():
    file_img = request.files.get('img', "")
    file_video = request.files.get('video', None)
    file_audio = request.files.get('audio', None)
    ids = '{0:%Y%m%d%H%M%S%f}'.format(datetime.datetime.now())
    status = {'status': 'faile', 'message':'no file data'}
    if file_video:
        filename = secure_filename(file_video.filename)
        file_path = os.path.join(app.config['DATA_FOLDER'], filename)
        file_video.save(file_path)
        video = VideoFileClip(file_path)
        audio = video.audio
        voc_path = os.path.join(app.config['DATA_FOLDER'], ids[:-1] + '.wav')
        audio.write_audiofile(voc_path)
    elif file_audio:
        voc_path = os.path.join(app.config['DATA_FOLDER'], ids[:-1] + '.wav')
        file_audio.save(voc_path)
    else:
        return jsonify(status), 200

    if file_img:
        img_path = os.path.join(app.config['DATA_FOLDER'], ids + '.png')
        file_img.save(img_path)
        try:
            status = do_search(img_path, voc_path)
            status[1] = request.url_root + "data/" + str(status[1]) + '.png'
        except:
            status = {'status': 'faile', 'message':'please confirm only one face in camera'}
            return jsonify(status), 200
        finally:
            os.remove(img_path)
            os.remove(voc_path)
            return "{}".format(status), 200
    else:
        return jsonify(status), 200
    # return jsonify(status), 200


if __name__ == "__main__":
    app.run(ssl_context='adhoc',host="0.0.0.0",port=5000)
