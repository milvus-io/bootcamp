import os
from flask import Flask, request
from flask_cors import CORS
from flask_restful import reqparse
from werkzeug.utils import secure_filename
from common.config import UPLOAD_FOLDER
from common.common import allowed_file
from controller.video import list_video
from controller.video import upload_video, delete_video, search_video
from controller.video import process_status
from . import model, sess, graph
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CORS(app)

@app.route('/v1/video', methods=['POST'])
def upload_video_api():
    file = request.files.get('file', "")
    if not file:
        return "no file data", 400
    if not file.name:
        return "need file name", 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(UPLOAD_FOLDER, filename))
        return upload_video(filename, model, sess, graph)
    return "file type not allowed", 413


@app.route('/v1/video', methods=['DELETE'])
def delete_video_api():
    args = reqparse.RequestParser(). \
        add_argument("ID", type=str). \
        parse_args()
    id = args["ID"]
    return delete_video(id)


@app.route('/v1/search', methods=['POST'])
def search_video_api():
    args = reqparse.RequestParser(). \
        add_argument("Num", type=int, required=True). \
        parse_args()
    num = args['Num']
    file = request.files.get('file', "")
    if not file:
        return "no file data", 400
    if not file.name:
        return "need file name", 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(UPLOAD_FOLDER, filename))
        return search_video(filename, num, model, sess, graph)
    return "file type not allowed", 413


@app.route('/v1/status', methods=['POST'])
def video_status_api():
    args = reqparse.RequestParser(). \
        add_argument("ID", type=str). \
        parse_args()
    id = args["ID"]
    return process_status(id)


@app.route('/v1/video', methods=['GET'])
def list_video_api():
    args = reqparse.RequestParser(). \
        add_argument("Reverse", type=bool, default=True). \
        add_argument("PageNum", type=int, default=0). \
        add_argument("PerPageCount", type=int, default=50). \
        parse_args()

    reverse = args["Reverse"]
    page = args["PageNum"]
    count = args['PerPageCount']

    if page < 0:
        page = 0
    if count < 0:
        count = 50
    return list_video(count, page, reverse)


@app.route('/v1/ping', methods=['GET'])
def ping_api():
    return "Pong"
